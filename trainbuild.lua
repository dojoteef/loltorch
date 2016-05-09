local dataset = require('dataset')
local nn = require('nn')
local nngraph = require('nngraph')
local optim = require('optim')
local path = require('pl.path')
local torch = require('torch')
require('TemporalBatchNormalization')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a model which predicts which champion builds are most successful')
cmd:text()
cmd:text('Options')
cmd:option('-batchsize',128,'how many inputs per batch')
cmd:option('-depth',3,'model has 6*depth+2 layers')
cmd:option('-epochs',200,'how many epochs to process')
cmd:option('-seed',1,'manually set the random seed')
cmd:option('-datadir','dataset','the directory where the dataset is located')
cmd:option('-snapshot','snapshot.t7','where to save snapshots of the in training model')
cmd:option('-model','model.t7','where to save the final model')
cmd:option('-modeltype','resnet','what type of model to create')
cmd:option('-continue','','where to load a previous model from to continue training')
cmd:option('-threshold',1e-6,'threshold to reach in training')
cmd:option('-debug',false,'whether to set debug mode')
cmd:text()

local params = cmd:parse(arg)
if params.debug then
    print('setting debug mode')
    nngraph.setDebug(true)
end


local dataLoader = dataset.Loader(params.datadir, 8, 1, 1)
local dataTrain, dataValidate, dataTest = dataLoader:load(params.batchsize)

local function collectAllGarbage()
    local timer = torch.Timer()
    print('collecting garbage...')
    local mem
    while mem ~= collectgarbage('count') do
        mem = collectgarbage('count')
        collectgarbage()
    end
    print('garbage collection complete - timer: '..timer:time().real)
end

local function calculateOutputFeatures(inputFrames, window, stride)
    return math.floor((inputFrames - window) / stride + 1)
end

local function getResnetModel()
    local function addResidualLayer(input, features, outputFrameSize, stride)
        local inputFrameSize
        local findConvolutions = function(node)
            -- when doing dfs, last convolution will be the most recently added
            -- convolution which is what we want to match output size of
            local convolutions = node.data.module:findModules('nn.TemporalConvolution')
            for _, convolution in ipairs(convolutions) do
                inputFrameSize = convolution.outputFrameSize
            end
        end
        input:dfs(findConvolutions)
        assert(inputFrameSize, 'Could not find outputFrameSize from previous TemporalConvolution module')

        stride = stride or 1
        features = calculateOutputFeatures(features, 1, stride)

        local layer = nn.TemporalConvolution(inputFrameSize, outputFrameSize, 1, stride)(input)
        layer = nn.TemporalBatchNormalization(features)(layer)
        layer = nn.ReLU(true)(layer)
        layer = nn.TemporalConvolution(outputFrameSize, outputFrameSize, 1, 1)(layer)
        layer = nn.TemporalBatchNormalization(features)(layer)

        local shortcut = input
        if stride > 1 then
            shortcut = nn.TemporalMaxPooling(1, stride)(shortcut)
        end
        if outputFrameSize > inputFrameSize then
            shortcut = nn.Padding(2, outputFrameSize - inputFrameSize, 2)(shortcut)
        end

        layer = nn.CAddTable()({layer, shortcut})
        layer = nn.ReLU(true)(layer)

        return layer, features
    end

    local inputCount = dataTrain.inputs:size(2)
    local features = calculateOutputFeatures(inputCount, 1, 2)

    local input = nn.Identity()()
    local model = nn.TemporalConvolution(dataLoader:embeddingSize(), 16, 1, 2)(input)
    model = nn.TemporalBatchNormalization(features)(model)
    model = nn.ReLU(true)(model)

    for _=1,params.depth do model, features = addResidualLayer(model, features, 16) end

    model, features = addResidualLayer(model, features, 32, 16, 2)
    for _=1,params.depth-1 do model, features = addResidualLayer(model, features, 32) end

    model, features = addResidualLayer(model, features, 64, 32, 2)
    for _=1,params.depth-1 do model, features = addResidualLayer(model, features, 64) end

    local view = nn.View(64)
    view:setNumInputDims(2)

    local targetSize = torch.LongTensor(dataTrain.targets:size())
    targetSize = targetSize[{{2, targetSize:size()[1]}}]
    model = nn.Linear(64, torch.prod(targetSize))(model)

    targetSize = targetSize:totable()
    local outputView = nn.View(table.unpack(targetSize))
    outputView:setNumInputDims(#targetSize)
    model = outputView(model)

    model = nn.gModule({input}, {model})
    model.verbose = params.debug
    model.name = 'resnet'

    local criterion = nn.ParallelCriterion(true)
    for _=1,targetSize[1] do
        criterion:add(nn.CosineEmbeddingCriterion())
    end

    return model, criterion
end

local function getMLPModel()
    local function getTrainingModel(input)
        local inputSplit = nn.SplitTable(1, 2)(input)

        local target = nn.Identity()()
        local targetSplit = nn.SplitTable(1, 2)(target)

        local numElements = dataTrain.targets:size(2)
        local splitInputs = {inputSplit:split(numElements)}
        local splitTargets = {targetSplit:split(numElements)}

        local output = {}
        for i=1,numElements do
            table.insert(output, nn.Identity()({splitInputs[i], splitTargets[i]}))
        end

        local model = nn.gModule({input, target}, output)

        local criterion = nn.ParallelCriterion(true)
        for _=1,dataTrain.targets:size(2) do
            criterion:add(nn.CosineEmbeddingCriterion())
        end

        local optimState = {
            momentum = 0.9,
            dampening = 0,
            nesterov = true,
            regime = {
                [1]={learningRate=1e-1,weightDecay=1e-4},
                [80]={learningRate=1e-2},
                [120]={learningRate=1e-3,weightDecay=0},
            }
        }

        return {model=model,criterion=criterion,optimState=optimState}
    end

    local hidden = 100
    local inputCount = dataTrain.inputs:size(2)
    local features = calculateOutputFeatures(inputCount, 1, 2)

    local input = nn.Identity()()
    local model = nn.TemporalConvolution(dataLoader:embeddingSize(), 16, 1, 2)(input)
    model = nn.TemporalBatchNormalization(features)(model)

    local inputSize = torch.LongTensor(dataTrain.inputs:size())
    inputSize = inputSize[{{2, inputSize:size()[1]}}]:totable()
    local inputView = nn.View(features*16)
    inputView:setNumInputDims(#inputSize)
    model = inputView(model)

    model = nn.Linear(features*16, hidden)(model)
    model = nn.ReLU(true)(model)

    local targetSize = torch.LongTensor(dataTrain.targets:size())
    targetSize = targetSize[{{2, targetSize:size()[1]}}]
    model = nn.Linear(hidden, torch.prod(targetSize))(model)

    targetSize = targetSize:totable()
    local outputView = nn.View(table.unpack(targetSize))
    outputView:setNumInputDims(#targetSize)
    model = outputView(model)

    model = nn.gModule({input}, {model})
    model.name = 'mlp'

    return model, getTrainingModel(model())
end

local modelFactory = {
    resnet=getResnetModel,
    mlp=getMLPModel
}

local function getModel()
    if not path.exists(params.continue) then
        print('creating model')
        local factoryMethod = modelFactory[params.modeltype]
        return factoryMethod()
    else
        print('continuing from '..params.continue)
        local loaded = torch.load(params.continue)
        assert(loaded and loaded.model and loaded.training, 'Unable to load '..params.continue)
        return loaded.model, loaded.training
    end
end

local function evaluateModel(model, data)
    print('Evaluating...')
    model:evaluate()

    local losses = {}
    local criterion = nn.MSECriterion()
    while data:hasNextBatch() do
        local inputs, targets = data:nextBatch()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        table.insert(losses, loss)
    end

    losses = torch.Tensor(losses)
    local loss = torch.sum(losses)/torch.numel(losses)
    print('loss: '..loss)

    return math.abs(loss)
end

local function trainModel(model, training)
    local criterion = training.criterion
    local trainingModel = training.model
    local parameters, parameterGradients = trainingModel:getParameters()
    local function calculateLoss(inputParameters)
        if parameters ~= inputParameters then
            parameters:copy(inputParameters)
        end

        parameterGradients:zero()

        local inputs, targets, outcomes = dataTrain:nextBatch()
        local predictions = trainingModel:forward({inputs, targets})
        local loss = criterion:forward(predictions, outcomes)
        local dloss_dw = criterion:backward(predictions, outcomes)
        trainingModel:backward(inputs, dloss_dw)

        return loss, parameterGradients
    end

    local timer = torch.Timer()
    local optimState = training.optimState
    local snapshot = {model=model,training=training}
    local startingEpoch = optimState.epoch or 1
    for epoch=startingEpoch, params.epochs do
        local regimeChange = optimState.regime[epoch]
        if regimeChange then
            for k,v in pairs(regimeChange) do
                optimState[k] = v
            end
        end

        optimState.epoch = epoch
        print('starting epoch '..epoch..' - timer: '..timer:time().real)

        model:training()
        trainingModel:training()
        local losses = {}
        while dataTrain:hasNextBatch() do
            local _, batchLoss = optim.sgd(calculateLoss, parameters, optimState)
            table.insert(losses, batchLoss[1])
        end

        losses = torch.Tensor(losses)
        local loss = torch.sum(losses)/torch.numel(losses)
        print('training loss: '..loss)
        torch.save(params.snapshot, snapshot)

        if evaluateModel(model, dataValidate) < params.threshold then
            break
        end

        dataTrain:resetBatch()
        dataValidate:resetBatch()

        -- collect all garbage at the end of each epoch because my memory
        -- constraints are tight
        collectAllGarbage()
    end

    return model
end

local function errorHandler(model)
    local function handler(errmsg)
        errmsg = errmsg..'\n'..debug.traceback()
        print(errmsg)

        if params.debug then
            local svg = model.name..'.svg'
            if not path.exists(svg) then
                nngraph.display(model)
            else
                os.execute('open -a Safari '..svg)
            end

            debug.debug()
        end

        return errmsg
    end

    return handler
end

collectAllGarbage()
local model, training = getModel()
local ok, errmsg = xpcall(trainModel, errorHandler(model), model, training)
if not ok then
    print('training failed!')

    print(errmsg)
    os.exit()
end

model:evaluate()
torch.save(params.model, model)

ok, errmsg = xpcall(evaluateModel, errorHandler(model), model, dataTest)
if not ok then
    print('testing failed!')

    print(errmsg)
    os.exit()
end
