local dataset = require('dataset')
local dir = require('pl.dir')
local models = require('models')
local nngraph = require('nngraph')
local opts = require('opts')
local path = require('pl.path')
local snapshot = require('snapshot')
local torch = require('torch')
local train = require('train')
local utils = require('utils')

local debugModel
local function getErrorHandler(debugOnError)
    local function handler(errmsg)
        errmsg = errmsg..'\n'..debug.traceback()
        print(errmsg)

        if debugOnError then
            if debugModel then
                local svg = debugModel.name..'.svg'
                if not path.exists(svg) then
                    nngraph.display(debugModel)
                else
                    os.execute('open -a Safari '..svg)
                end
            end

            debug.debug()
        end

        return errmsg
    end

    return handler
end

local tester = {}
function tester:init(dataLoader, margin, snapshotDir, threadCount)
    self.dataLoader = dataLoader
    self.dir = snapshotDir
    self.margin = margin
    self.threads = threadCount
end

function tester:calculateLoss(fold, best)
    local latest = snapshot.Snapshot(self.dir, fold)
    latest:load(true)

    local model = latest.model
    local criterion = train.setupCriterion(self.dataLoader, self.margin)
    local trainer = train.setupTrainer(self.dataLoader, model)

    local evalProgress = {}
    evalProgress.action = 'Test'

    local loss = train.evaluate(trainer, criterion, self.dataLoader.testData, evalProgress, self.threads)
    if best and loss < best.loss then
        best.loss = loss
        best.model = model
        evalProgress.best = true
    end

    evalProgress.flush = true
    utils.updateProgress(evalProgress)

    return loss
end

local function trainBuild(opt)
    -- Load up the current progress
    local progress = snapshot.Progress(opt)
    progress:load()

    -- Setup the data loader
    local kfolds = progress.kfolds
    local dataLoader = dataset.Loader(opt.datadir, progress.testratio, kfolds, opt.batchsize)

    -- Setup the model loader
    local modelLoader = models.Loader(dataLoader, opt)

    -- Setup the tester
    tester:init(dataLoader, progress.margin, progress.dir, opt.threads)

    -- Only do one set of folds if we aren't doing cross validation
    kfolds = opt.cross and kfolds or 1
    for fold=progress.fold,kfolds do
        print('Training fold #'..fold..' of '..kfolds)

        progress:save()
        utils.collectAllGarbage()

        local latest = snapshot.Snapshot(progress.dir, fold)
        latest:load()
        modelLoader:load(latest)
        debugModel = latest.model

        local trainer = train.Trainer(fold, latest, dataLoader, opt)
        trainer:train(progress.margin, opt.strategy, progress.frequency)

        tester:calculateLoss(fold)
        progress:advance()
    end

    -- Final save
    progress:save()

    print('Testing')
    utils.progressHeader()

    local best = {loss=math.huge}
    local testingLoss = 0
    for fold=1,kfolds do
        testingLoss = testingLoss + tester:calculateLoss(fold, best)
    end

    testingLoss = testingLoss/kfolds
    print('\nAverage Loss: '..testingLoss)

    -- Ensure path exists
    dir.makepath(opt.models)

    -- Make sure the model is in evaluation mode
    best.model:evaluate()

    -- Save the model
    local modelfile = path.join(opt.models, opt.model)
    torch.save(modelfile, best.model)
end

local opt = opts.parse(arg)
local errorHandler = getErrorHandler(opt.debug)
local ok, errmsg = xpcall(trainBuild, errorHandler, opt)
if not ok then
    print('Training failed!')

    print(errmsg)
    os.exit(-1)
end
