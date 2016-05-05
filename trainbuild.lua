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
cmd:option('-model','model.t7','where to save/load the model to/from (if model exists it will be loaded and training will continue on it)')
cmd:option('-modeltype','resnet','what type of model to create')
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

local function getResnetModel()
    local function calcluateOutputFeatures(inputFrames, window, stride)
        return (inputFrames - window) / stride + 1
    end

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
        features = calcluateOutputFeatures(features, 1, stride)

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

    local inputSize = dataLoader.mappings.inputs.size
    local inputCount = dataTrain.inputs:size(2)
    local targetCount = dataTrain.targets:size(2)
    local embeddingSize = 3

    local input = nn.Identity()()
    local model = nn.LookupTable(inputSize, embeddingSize)(input)
    model = nn.TemporalConvolution(embeddingSize, 16, 1, 2)(model)

    local features = calcluateOutputFeatures(inputCount, 1, 2)
    model = nn.TemporalBatchNormalization(features)(model)
    model = nn.ReLU(true)(model)

    for _=1,params.depth do model, features = addResidualLayer(model, features, 16) end

    model, features = addResidualLayer(model, features, 32, 16, 2)
    for _=1,params.depth-1 do model, features = addResidualLayer(model, features, 32) end

    model, features = addResidualLayer(model, features, 64, 32, 2)
    for _=1,params.depth-1 do model, features = addResidualLayer(model, features, 64) end

    local view = nn.View(64)
    view:setNumInputDims(2)
    model = view(model)
    model = nn.Linear(64, targetCount)(model)

    model = nn.gModule({input}, {model})
    model.name = 'resnet'

    return model, nn.MSECriterion()
end

local function getMLPModel()
    local hidden = 100
    local inputSize = dataTrain.inputs:size(2)
    local targetSize = dataTrain.targets:size(2)

    local input = nn.Identity()()
    local model = nn.Reshape(inputSize, true)(input)
    model = nn.BatchNormalization(inputSize)(model)
    model = nn.Linear(inputSize, hidden)(model)
    model = nn.Linear(hidden, targetSize)(model)

    model = nn.gModule({input}, {model})
    model.name = 'mlp'

    return model, nn.MSECriterion()
end

local modelFactory = {
    resnet=getResnetModel,
    mlp=getMLPModel
}

local function getModel()
    local model, criterion
    if not path.exists(params.model) then
        local factoryMethod = modelFactory[params.modeltype]
        model, criterion = factoryMethod()
    else
        print('loading model from '..params.model)
        model = torch.load(params.model)
    end

    assert(model, 'Unable to load model')
    return model, criterion
end

local function evaluateModel(model, data, criterion)
    print('Evaluating...')
    model:evaluate()

    local losses = {}
    while data:hasNextBatch() do
        local inputs, targets = data:nextBatch()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        table.insert(losses, loss)
    end

    losses = torch.Tensor(losses)
    local loss = torch.sum(losses)/(torch.numel(losses)*data:targetScale())
    print('loss: '..loss)

    return math.abs(loss)
end

local function trainModel(model, criterion)
    local optimState = {
        learningRate = 1e-4,
        weightDecay = 1e-4,
        momentum = 0.9,
        dampening = 0,
        nesterov = true
    }

    local regime = {
        [1]={learningRate=1e-4,weightDecay=5e-4},
        [3]={learningRate=5e-5},
        [30]={learningRate=1e-5,weightDecay=0},
        [44]={learningRate=5e-6},
        [53]={learningRate=1e-6},
    }

    local parameters, parameterGradients = model:getParameters()
    local function calculateLoss(inputParameters)
        if parameters ~= inputParameters then
            parameters:copy(inputParameters)
        end

        parameterGradients:zero()

        local inputs, targets = dataTrain:nextBatch()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        local dloss_dw = criterion:backward(predictions, targets)
        model:backward(inputs, dloss_dw)

        return loss, parameterGradients
    end

    for epoch=1, params.epochs do
        local regimeChange = regime[epoch]
        if regimeChange then
            optimState.learningRate = regimeChange.learningRate or optimState.learningRate
            optimState.weightDecay = regimeChange.weightDecay or optimState.weightDecay
        end

        optimState.epoch = epoch
        print('starting epoch '..epoch..':')

        model:training()
        local losses = {}
        while dataTrain:hasNextBatch() do
            local _, batchLoss = optim.sgd(calculateLoss, parameters, optimState)
            table.insert(losses, batchLoss[1])
        end

        losses = torch.Tensor(losses)
        local loss = torch.sum(losses)/(torch.numel(losses)*dataTrain:targetScale())
        print('training loss: '..loss)
        torch.save(params.snapshot, model)

        if evaluateModel(model, dataValidate, criterion) < params.threshold then
            break
        end

        dataTrain:resetBatch()
        dataValidate:resetBatch()
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

local model, criterion = getModel()
local ok, errmsg = xpcall(trainModel, errorHandler(model), model, criterion)
if not ok then
    print('training failed!')

    print(errmsg)
    os.exit()
end

model:evaluate()
torch.save(params.model, model)

ok, errmsg = xpcall(evaluateModel, errorHandler(model), model, dataTest, criterion)
if not ok then
    print('testing failed!')

    print(errmsg)
    os.exit()
end
