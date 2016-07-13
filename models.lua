local nn = require('nn')
local torch = require('torch')

local models = {}
local _models = torch.class('Models.Loader', models)

local function getModel(dataLoader, layers, opt)
    local inputCount = dataLoader.inputs:size(2)
    local inputEmbedding = dataLoader.inputs:size(3)

    local targetCount = dataLoader.targets:size(2)
    local targetEmbedding = dataLoader.targets:size(3)

    local input = nn.Identity()()
    local inputSplit = nn.SplitTable(1, 2)(input)
    local splitInputs = {inputSplit:split(inputCount)}

    local dropout = opt.dropout

    local output = {}
    for i=1,inputCount do
        local layerInput = inputEmbedding
        local model = nn.Identity()(splitInputs[i])
        for l=1,#layers do
            local layer = layers[l]
            local layerOutput = layer.count
            local layerModel = model
            local transfer = layer.transfer
            local transform = layer.transform or nn.Linear

            local depthInput = layerInput
            local depth = layer.depth or 1
            for d=1,depth do
                model = transform(depthInput, layerOutput)(model)

                if dropout then
                    model = nn.Dropout(math.min(dropout.max, dropout.min + (l-1)*dropout.gain))(model)
                end

                if transfer then
                    local func = transfer.func
                    local params = transfer.params or {}
                    if depth == 1 or d < depth then
                        model = func(table.unpack(params))(model)
                    end
                end

                depthInput = layerOutput
            end

            if depth > 1 then
                local shortcut
                if layerInput ~= layerOutput then
                    shortcut = transform(layerInput, layer.count)(layerModel)
                else
                    shortcut = nn.Identity()(layerModel)
                end

                model = nn.CAddTable()({model, shortcut})

                if transfer then
                    local func = transfer.func
                    local params = transfer.params or {}
                    model = func(table.unpack(params))(model)
                end
            end

            layerInput = layerOutput
        end

        table.insert(output, model)
    end

    local gates = opt.gates
    if gates then
        for i=1,inputCount do
            local gaterInput = inputEmbedding
            local gate = nn.Identity()(splitInputs[i])
            for g=1,#gates do
                local gater = gates[g]
                local gaterOutput = gater.count

                local transform = gater.transform or nn.Linear
                gate = transform(gaterInput, gaterOutput)(gate)

                local transfer = gater.transfer
                if transfer then
                    local func = transfer.func
                    local params = transfer.params or {}
                    gate = func(table.unpack(params))(gate)
                end

                gaterInput = gaterOutput
            end

            output[i] = nn.CMulTable()({output[i], gate})
        end
    end

    output = nn.CAddTable()(output)

    local transform = opt.transform or nn.Linear
    local outputSize = layers[#layers].count
    output = transform(outputSize, targetCount * targetEmbedding)(output)

    local outputView = nn.View(targetCount, targetEmbedding)
    outputView:setNumInputDims(2)
    output = outputView(output)

    return nn.gModule({input}, {output})
end

local function getMLP192Model(dataLoader)
    local layers = {
        {count=192,transfer={func=nn.HardShrink,params={0.25}}},
    }

    local model = getModel(dataLoader, layers, {dropout={min=.4,max=.4,gain=0}})
    model.name = 'mlp192'

    return model
end

local function getMLP384Model(dataLoader)
    local layers = {
        {count=192,transfer={func=nn.HardShrink,params={0.25}}},
        {count=384,transfer={func=nn.HardShrink,params={0.25}}},
    }

    local model = getModel(dataLoader, layers, {dropout={min=.4,max=.5,gain=.1}})
    model.name = 'mlp384'

    return model
end

local function getGated192Model(dataLoader)
    local targetCount = dataLoader.targets:size(2)
    local layers = {
        {count=96,transfer={func=nn.ReLU,params={true}}},
        {count=192,transfer={func=nn.ReLU,params={true}}},
        {count=targetCount}
    }

    local gates = {
        {count=48,transfer={func=nn.ReLU,params={true}}},
        {count=96,transfer={func=nn.ReLU,params={true}}},
        {count=targetCount,transfer={func=nn.Tanh}}
    }

    local model = getModel(dataLoader, layers, {gates=gates, dropout={min=.4,max=.5,gain=.1}})
    model.name = 'gated192'

    return model
end

local function getGated384Model(dataLoader)
    local targetCount = dataLoader.targets:size(2)
    local layers = {
        {count=192,transfer={func=nn.ReLU,params={true}}},
        {count=384,transfer={func=nn.ReLU,params={true}}},
        {count=targetCount}
    }

    local gates = {
        {count=48,transfer={func=nn.ReLU,params={true}}},
        {count=96,transfer={func=nn.ReLU,params={true}}},
        {count=targetCount,transfer={func=nn.Tanh}}
    }

    local model = getModel(dataLoader, layers, {gates=gates, dropout={min=.4,max=.5,gain=.1}})
    model.name = 'gated384'

    return model
end

local function getGated768Model(dataLoader)
    local targetCount = dataLoader.targets:size(2)
    local layers = {
        {count=384,transfer={func=nn.ReLU,params={true}}},
        {count=768,transfer={func=nn.ReLU,params={true}}},
        {count=targetCount}
    }

    local gates = {
        {count=96,transfer={func=nn.ReLU,params={true}}},
        {count=192,transfer={func=nn.ReLU,params={true}}},
        {count=targetCount,transfer={func=nn.Tanh}}
    }

    local model = getModel(dataLoader, layers, {gates=gates, dropout={min=.4,max=.5,gain=.1}})
    model.name = 'gated768'

    return model
end

local function getCosine192Model(dataLoader)
    local targetCount = dataLoader.targets:size(2)
    local layers = {
        {count=96,transfer={func=nn.ReLU,params={true}},transform=nn.Cosine},
        {count=192,transfer={func=nn.ReLU,params={true}},transform=nn.Cosine},
        {count=targetCount},
    }

    local gates = {
        {count=48,transfer={func=nn.ReLU,params={true}}},
        {count=96,transfer={func=nn.ReLU,params={true}}},
        {count=targetCount,transfer={func=nn.Tanh}}
    }

    local model = getModel(dataLoader, layers, {gates=gates, dropout={min=.4,max=.5,gain=.1}})
    model.name = 'cosine192'

    return model
end

local function getCosine384Model(dataLoader)
    local targetCount = dataLoader.targets:size(2)
    local layers = {
        {count=192,transfer={func=nn.ReLU,params={true}},transform=nn.Cosine},
        {count=384,transfer={func=nn.ReLU,params={true}},transform=nn.Cosine},
        {count=targetCount},
    }

    local gates = {
        {count=48,transfer={func=nn.ReLU,params={true}}},
        {count=96,transfer={func=nn.ReLU,params={true}}},
        {count=targetCount,transfer={func=nn.Tanh}}
    }

    local model = getModel(dataLoader, layers, {gates=gates, dropout={min=.4,max=.5,gain=.1}})
    model.name = 'cosine384'

    return model
end

local modelFactory = {
    mlp192=getMLP192Model,
    mlp384=getMLP384Model,
    gated192=getGated192Model,
    gated384=getGated384Model,
    gated768=getGated768Model,
    cosine192=getCosine192Model,
    cosine384=getCosine384Model,
}

function _models:__init(dataLoader, opt)
    self.dataLoader = dataLoader
    self.modeltype = opt.modeltype
end

function _models:load(snapshot)
    if snapshot.model then
        print('Continuing from epoch #'..snapshot.epoch..'\n')
    else
        print('Creating model\n')
        local factoryMethod = modelFactory[self.modeltype]
        snapshot.model = factoryMethod(self.dataLoader)
    end

    snapshot.model.verbose = self.verbose
end

return models
