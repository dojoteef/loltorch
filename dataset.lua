local cjson = require('cjson')
require('dpnn')
local file = require('pl.file')
local nn = require('nn')
local path = require('pl.path')
local torch = require('torch')

local dataset = {
    maxSpellCount=2,
    maxItemCount = 7,
    maxRuneCount = 30,
    maxMasteryCount = 30
}
local _data = torch.class('Dataset.Data', dataset)
local _loader = torch.class('Dataset.Loader', dataset)

function _loader:__init(datadir, ptrain, pvalidate, ptest)
    assert(path.isdir(datadir), 'Invalid data directory specified')
    self.dir = datadir
    self.mappings = torch.load(self.dir..path.sep..'map.t7')

    local input = nn.Identity()()
    local inputOneHot = nn.OneHot(self.mappings.inputs.size)(input)
    self.inputOneHot = nn.gModule({input}, {inputOneHot})

    local data = file.read(self.dir..path.sep..'champions.json')
    local ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load champions.json')
    self.champions = json

    data = file.read(self.dir..path.sep..'items.json')
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load items.json')
    self.items = json

    data = file.read(self.dir..path.sep..'masteries.json')
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load masteries.json')
    self.masteries = json

    data = file.read(self.dir..path.sep..'runes.json')
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load runes.json')
    self.runes = json

    data = file.read(self.dir..path.sep..'spells.json')
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load spells.json')
    self.spells = json

    local total = ptrain + pvalidate + ptest
    self.trainRatio = ptrain / total
    self.validateRatio = pvalidate / total
    self.testRatio = ptest / total
end

function _loader:load(batchsize)
    local inputs = torch.load(self.dir..path.sep..'inputs.t7'):double()
    local targets = torch.load(self.dir..path.sep..'targets.t7'):double()
    assert(inputs:size(1) == targets:size(1), 'Dataset mismatch: input size ('..inputs:size(1)..'), target size ('..targets:size(1)..')')

    local count = inputs:size(1)
    local trainIndex = math.floor(count * self.trainRatio)
    local validateIndex = trainIndex + math.floor(count * self.validateRatio)

    local indices = torch.randperm(inputs:size(1)):long()
    local view = indices[{{1, trainIndex}}]
    local trainData = dataset.Data(self, inputs:index(1, view), targets:index(1, view), batchsize)

    view = indices[{{trainIndex+1, validateIndex}}]
    local validateData = dataset.Data(self, inputs:index(1, view), targets:index(1, view), batchsize)

    view = indices[{{validateIndex+1, count}}]
    local testData = dataset.Data(self, inputs:index(1, view), targets:index(1, view), batchsize)

    return trainData, validateData, testData
end

function _loader:getChampionIndex(champion)
    local championId
    for name, data in pairs(self.champions.data) do
        if name == champion then
            championId = data.id
            break
        end
    end

    assert(championId, 'Could not find champion: '..champion)
    return self.mappings.inputs.champions[championId]
end

function _loader:getLaneIndex(lane)
    return self.mappings.inputs.lanes[lane]
end

function _loader:getRoleIndex(role)
    return self.mappings.inputs.roles[role]
end

function _loader:getDataByIndex(index)
    for datatype, list in pairs(self.mappings.targets) do
        if type(list) == 'table' then
            for id, i in pairs(list) do
                if index == i then
                    local data = self[datatype].data
                    local value = data[tostring(id)]
                    local name = value and value.name

                    if not name then
                        -- must be by name
                        for k, v in pairs(data) do
                            if v.id == id then
                                name = k
                                break
                            end
                        end
                    end

                    return datatype, name
                end
            end
        end
    end
end

local function sampleOneHot(predictions, loader, count, targetCount, startIndex)
    for _=1,targetCount do
        local endIndex = startIndex+count-1
        local slice = predictions[{1, {startIndex,endIndex}}]

        local _,index = torch.max(slice, 1)
        local datatype, name = loader:getDataByIndex(index[1])
        if name then
            print('datatype: '..datatype..', name: '..name)
        end

        startIndex = endIndex+1
    end

    return startIndex
end

function _loader:sampleOneHot(model, input)
    local predictions = model:forward(self.inputOneHot:forward(input))

    local length = self.mappings.targets.size
    local nextIndex = sampleOneHot(predictions, self, length, dataset.maxSpellCount, 1)
    nextIndex = sampleOneHot(predictions, self, length, dataset.maxItemCount, nextIndex)
    nextIndex = sampleOneHot(predictions, self, length, dataset.maxRuneCount, nextIndex)
    sampleOneHot(predictions, self, length, dataset.maxMasteryCount, nextIndex)
end

function _loader:sample(model, input)
    local predictions = model:forward(self.inputOneHot:forward(input))
    predictions:ceil()

    for i=1,predictions:size(2) do
        local index = predictions[{1, i}]
        local datatype, name = self:getDataByIndex(index)
        if name then
            print('datatype: '..datatype..', name: '..name)
        end
    end
end

function _data:__init(loader, inputs, targets, batchsize)
    self.loader = loader
    self.inputs = inputs
    self.targets = targets

    local target = nn.Identity()()
    local targetOneHot = nn.OneHot(loader.mappings.targets.size)(target)
    targetOneHot = nn.Reshape(loader.mappings.targets.size * self.targets:size(2))(targetOneHot)
    self.targetOneHot = nn.gModule({target}, {targetOneHot})

    local inputSize = self.inputs:size():totable()
    inputSize[1] = batchsize

    local targetSize = self.targets:size():totable()
    targetSize[1] = batchsize

    self.batch = {
        inputs=torch.zeros(unpack(inputSize)),
        targets=torch.zeros(unpack(targetSize))
    }

    self:resetBatch()
end

function _data:batchSize()
    return self.batch.inputs:size(1)
end

function _data:hasNextBatch()
    return self.batchIndex < self.inputs:size(1)
end

function _data:resetBatch()
    self.batchIndex = 1
    self.indices =  torch.randperm(self.inputs:size(1)):long()
end

function _data:nextBatch()
    if self:hasNextBatch() then
        local endIndex = math.min(self.indices:size(1), self.batchIndex+self:batchSize()-1)
        local batchIndices = self.indices[{{self.batchIndex, endIndex}}]
        self.batchIndex = self.batchIndex + self:batchSize()

        self.batch.inputs:index(self.inputs, 1, batchIndices)
        self.batch.targets:index(self.targets, 1, batchIndices)

        return self.loader.inputOneHot:forward(self.batch.inputs), self.batch.targets--self.targetOneHot:forward(self.batch.targets)
    end
end

return dataset
