local nn = require('nn')
require('nngraph')
local optim = require('optim')
local threads = require('threads')
local tds = require('tds')
local torch = require('torch')
local utils = require('utils')

local train = {}
local _trainer = torch.class('Train.Trainer', train)

function _trainer:__init(fold, snapshot, dataLoader, opt)
    self.snapshot = snapshot
    self.dataLoader = dataLoader
    self.trainData, self.validateData = dataLoader:getFold(fold)

    self.epochs = opt.epochs
    self.patience = opt.patience
    self.threads = opt.threads
    self.verbose = opt.verbose
end

function _trainer:setup(margin, reload)
    if not self.criterion or reload then
        self.criterion = train.setupCriterion(self.dataLoader, margin)
    end

    if not self.trainer or reload then
        self.trainer = train.setupTrainer(self.dataLoader, self.snapshot.model)
        self.parameters, self.gradients = self.trainer:getParameters()

        if self.snapshot.epoch == 0 then
            self.parameters:uniform(-0.08, 0.08)
        end
    end
end

function train.setupTrainer(dataLoader, model)
    local inputModel = model()
    local inputSplit = nn.SplitTable(1, 2)(inputModel)

    local target = nn.Identity()()
    local targetSplit = nn.SplitTable(1, 2)(target)

    local numElements = dataLoader.targets:size(2)
    local splitInputs = {inputSplit:split(numElements)}
    local splitTargets = {targetSplit:split(numElements)}

    local output = {}
    for i=1,numElements do
        table.insert(output, nn.Identity()({splitInputs[i], splitTargets[i]}))
    end

    local trainer = nn.gModule({inputModel, target}, output)
    trainer.name = 'training.'..model.name

    return trainer
end

function train.setupCriterion(dataLoader, margin)
    local criterion = nn.ParallelCriterion(true)
    local numElements = dataLoader.targets:size(2)
    for _=1,numElements do
        criterion:add(nn.CosineEmbeddingCriterion(margin))
    end

    return criterion
end

function train.evaluate(model, criterion, data, progress, threadnum)
    model:evaluate()

    local totalLoss = 0
    local batchCount = data:batchCount()

    progress.index = 0
    progress.timer = torch.Timer()
    progress.batchCount = batchCount

    local pool = train.newPool(model, criterion, data, threadnum)
    for i=1,batchCount do
        pool:addjob(
            function(index)
                return _G.calculateLoss(index)
            end,
            function(loss)
                totalLoss = totalLoss + loss

                progress.loss = totalLoss
                progress.index = progress.index + 1
                utils.updateProgress(progress)
            end,
            i
        )
    end
    pool:synchronize()
    pool:terminate()

    local loss = totalLoss/batchCount
    return loss
end

function train.newPool(model, criterion, data, threadnum)
    threads.serialization('threads.sharedserialize')

    local rngState = torch.getRNGState()
    local pool = threads.Threads(
        threadnum,
        function ()
            require('dataset')
            require('nn')
            require('nngraph')
            require('tds')
            _G.threads = require('threads')
            _G.torch = require('torch')
            _G.utils = require('utils')
        end,

        function ()
            local threadCriterion = criterion:clone()
            local threadModel = model:clone('weight', 'bias')
            local _, threadGradients = threadModel:parameters()
            threadGradients = threadModel.flatten(threadGradients)

            _G.torch.setRNGState(rngState)
            _G.utils.collectAllGarbage()

            _G.calculateLoss = function(batchIndex)
                local inputs, targets, outcomes = data:getBatch(batchIndex)
                local modelInput = {inputs, targets}

                local predictions = threadModel:forward(modelInput)
                local loss = threadCriterion:forward(predictions, outcomes)
                return loss, predictions
            end

            _G.calculateGradient = function(batchIndex, predictions)
                threadGradients:zero()

                local inputs, targets, outcomes = data:getBatch(batchIndex)
                local modelInput = {inputs, targets}

                local dloss_dw = threadCriterion:backward(predictions, outcomes)
                threadModel:backward(modelInput, dloss_dw)

                -- Clip gradients element-wise
                threadGradients:clamp(-5, 5)

                return threadGradients
            end
        end
    )

    return pool
end

local function getLearningParameters(strategy)
    local regimes
    local optimState
    if strategy <= 1 then
        optimState = {
            initialMomentum = 0.5,
            momentumGrowth=2e-7,
            dampening = 0,
            nesterov = true,
            learningRate=1e-2,
            learningRateDecay=1e-7,
        }
        regimes = {
            {learningRate=5e-3,initialMomentum=0.7},
            {learningRate=1e-3,initialMomentum=0.9},
            {learningRate=1e-4,initialMomentum=0.999},
        }
    elseif strategy <= 2 then
        optimState = {
            initialMomentum = 0.7,
            momentumGrowth=6e-7,
            dampening = 0,
            nesterov = true,
            learningRate=5e-1,
            learningRateDecay=1e-6,
        }
        regimes = {
            {learningRate=5e-3,initialMomentum=0.9},
            {learningRate=1e-3,initialMomentum=0.95},
            {learningRate=1e-4,initialMomentum=0.999},
        }
    elseif strategy >= 3 then
        optimState = {
            momentum = 0.9,
            dampening = 0,
            nesterov = true,
            learningRate=1e-1,
            learningRateDecay=0,
        }
        regimes = {
            {learningRate=1e-2,momentum=0.95},
            {learningRate=5e-3,momentum=0.995},
            {learningRate=1e-3,momentum=0.999},
        }
    end

    return optimState, {epoch=0,index=0,regimes=regimes}
end

local function printRegime(prefix, regime)
    local attributes = {}
    for k,v in pairs(regime) do
        if type(v) == 'number' then
            table.insert(attributes, tostring(k)..'='..tostring(v))
        end
    end
    print(prefix..table.concat(attributes, ', '))
end

function _trainer:trainEpoch(epoch, optimState, updateFrequency)
    local trainData = self.trainData

    -- Make sure the model is in training mode and the data is shuffled
    self.trainer:training()
    trainData:shuffle()

    local batchCount = trainData:batchCount()
    local progress = {}
    progress.epoch = epoch
    progress.action = 'Train'
    progress.loss = 0
    progress.index = 0
    progress.batchCount = batchCount
    progress.timer = torch.Timer()

    local parameters = self.parameters
    local function updateParameters(loss, gradients)
        if optimState.momentumGrowth then
            local nevals = optimState.evalCounter or 0
            local momentum = optimState.initialMomentum
            optimState.momentum = math.min(.999999, momentum * (1+nevals*optimState.momentumGrowth))
        end

        optim.sgd(function() return loss, gradients end, parameters, optimState)
    end

    local updateLosses = tds.Vec()
    local updateGradients = tds.Vec()

    -- Need to preallocate the size of the vectors
    -- such that no realloc occurs in the threads
    -- as it seems that could cause incorrect memory
    -- semantics since there is no way to synchronize
    -- the realloc across threads in tds.Vec
    for i=1,updateFrequency do
        updateLosses[i] = 0
        updateGradients[i] = self.gradients:clone()
    end

    local batchIndex = tds.AtomicCounter()
    local updates = tds.AtomicCounter()
    local updateMutex = threads.Mutex()
    local updateMutexId = updateMutex:id()
    local updateCondition = threads.Condition()
    local updateConditionId = updateCondition:id()

    local pool = train.newPool(self.trainer, self.criterion, trainData, self.threads)
    for i=1,batchCount do
        pool:addjob(
            function()
                -- Unfortunately to ensure deterministic training this synchronization
                -- is needed, otherwise the model parameters could be changing in the
                -- middle of a gradient update, which has undefined behavior.  This
                -- approach allows for completely deterministic results each time
                -- given the same command line options allowing comparisons between
                -- models and free parameters (useful together with cross validation).
                local mutex = _G.threads.Mutex(updateMutexId)
                local condition = _G.threads.Condition(updateConditionId)

                mutex:lock()
                while updates:get() >= updateFrequency do
                    condition:wait(mutex)
                end

                local index = batchIndex:inc() + 1
                local updateIndex = (index-1) % updateFrequency + 1

                updates:inc()
                mutex:unlock()

                local loss, predictions = _G.calculateLoss(index)
                updateLosses[updateIndex] = loss

                local gradients = _G.calculateGradient(index, predictions)
                updateGradients[updateIndex]:copy(gradients)

                return loss
            end,
            function(loss)
                updateMutex:lock()
                local index = progress.index + 1
                local updateCount = updates:get()
                if index % updateFrequency == 0 or index == batchCount then
                    for t=1,updateCount do
                        updateParameters(updateLosses[t], updateGradients[t])
                    end

                    updates:set(0)

                    -- Only signal after doing all the parameter updates
                    for _=1,updateCount do
                        updateCondition:signal()
                    end
                end
                updateMutex:unlock()

                progress.index = index
                progress.loss = progress.loss + loss
                utils.updateProgress(progress)
            end,
            i
        )
    end
    pool:synchronize()
    pool:terminate()

    progress.flush = true
    progress.index = batchCount
    utils.updateProgress(progress)

    updateMutex:free()
    updateCondition:free()
end

function _trainer:processEpoch(epoch, margin, frequency)
    local regime = self.snapshot.regime
    local optimState = self.snapshot.optimState
    if epoch - regime.epoch > self.patience then
        regime.index = regime.index + 1
        local regimeChange = regime.regimes[regime.index]
        if not regimeChange then
            return
        end

        printRegime('\nChanging regime: ', regimeChange)
        utils.progressHeader()

        self.snapshot:load(true)
        self:setup(margin, true)
        for k,v in pairs(regimeChange) do
            optimState[k] = v
        end

        regime.epoch = self.snapshot.best.epoch
        epoch = regime.epoch + 1

        utils.collectAllGarbage()
    end

    self:trainEpoch(epoch, optimState, frequency)

    local progress = {}
    progress.epoch = epoch
    progress.action = 'Validate'

    local snapshot = self.snapshot
    local best = snapshot.best
    local loss = train.evaluate(self.trainer, self.criterion, self.validateData, progress, self.threads)
    if loss < best.loss then
        best.loss = loss
        best.epoch = epoch
        regime.epoch = epoch
        progress.best = true
    end

    progress.flush = true
    utils.updateProgress(progress)

    snapshot:update(best, epoch, regime, optimState)
    snapshot:save()

    return epoch + 1
end

function _trainer:train(margin, strategy, frequency)
    local defaultOptimState, defaultRegime = getLearningParameters(strategy)
    local regime = self.snapshot.regime or defaultRegime
    local optimState = self.snapshot.optimState or defaultOptimState
    self.snapshot:update(self.snapshot.best, self.snapshot.epoch, regime, optimState)

    local epoch = self.snapshot.epoch + 1
    printRegime(epoch == 1 and 'Initial regime: ' or 'Current regime: ', optimState)

    self:setup(margin)
    utils.progressHeader()
    while epoch and epoch <= self.epochs do
        epoch = self:processEpoch(epoch, margin, frequency)
        utils.collectAllGarbage()
    end

    print('Fold training completed\n')
end

return train
