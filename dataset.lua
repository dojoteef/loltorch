local cjson = require('cjson')
local file = require('pl.file')
local path = require('pl.path')
local torch = require('torch')

local dataset = {
    maxSpellCount=2,
    maxItemCount = 7,
    maxRuneCount = 30,
    maxMasteryCount = 30,

    -- Using the knowledge from this forum post for
    -- determining version, i.e. just using major/minor
    -- https://developer.riotgames.com/discussion/community-discussion/show/FJT1rYsF
    versionFromString = function(versionString)
        local _,_,version = string.find(versionString,"^(%d+%.%d+)")
        return 'version'..version
    end
}
local targetCount = 0
for _,count in pairs(dataset) do
    if type(count) == 'number' then
        targetCount = targetCount + count
    end
end
dataset.targetCount = targetCount

local _data = torch.class('Dataset.Data', dataset)
local _loader = torch.class('Dataset.Loader', dataset)

function _loader:__init(datadir, ptrain, pvalidate, ptest)
    assert(path.isdir(datadir), 'Invalid data directory specified')
    self.dir = datadir
    self.embeddings = torch.load(self.dir..path.sep..'embeddings.t7')

    local data = file.read(self.dir..path.sep..'champions.json')
    local ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load champions.json')
    self.champion = json

    data = file.read(self.dir..path.sep..'items.json')
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load items.json')
    self.item = json

    data = file.read(self.dir..path.sep..'masteries.json')
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load masteries.json')
    self.mastery = json

    data = file.read(self.dir..path.sep..'runes.json')
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load runes.json')
    self.rune = json

    data = file.read(self.dir..path.sep..'spells.json')
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load spells.json')
    self.spell = json

    local total = ptrain + pvalidate + ptest
    self.trainRatio = ptrain / total
    self.validateRatio = pvalidate / total
    self.testRatio = ptest / total
end

function _loader:embeddingSize()
    local _,embedding = next(self.embeddings)
    return embedding:size(1)
end

function _loader:load(batchsize)
    local inputs = torch.load(self.dir..path.sep..'inputs.t7')
    local targets = torch.load(self.dir..path.sep..'targets.t7')
    local outcomes = torch.load(self.dir..path.sep..'outcomes.t7')
    assert(inputs:type() == targets:type(), 'Dataset mismatch: input type ('..inputs:type()..'), target type ('..targets:type()..')')
    assert(inputs:size(1) == targets:size(1), 'Dataset mismatch: input size ('..inputs:size(1)..'), target size ('..targets:size(1)..')')
    assert(inputs:size(1) == outcomes:size(1), 'Dataset mismatch: input size ('..inputs:size(1)..'), outcome size ('..outcomes:size(1)..')')

    local count = inputs:size(1)
    local trainIndex = math.floor(count * self.trainRatio)
    local validateIndex = trainIndex + math.floor(count * self.validateRatio)

    local indices = torch.randperm(inputs:size(1)):long()
    local view = indices[{{1, trainIndex}}]
    local trainData = dataset.Data(self, inputs:index(1, view), targets:index(1, view), outcomes:index(1, view), batchsize)

    view = indices[{{trainIndex+1, validateIndex}}]
    local validateData = dataset.Data(self, inputs:index(1, view), targets:index(1, view), outcomes:index(1, view), batchsize, true)

    view = indices[{{validateIndex+1, count}}]
    local testData = dataset.Data(self, inputs:index(1, view), targets:index(1, view), outcomes:index(1, view), batchsize, true)

    return trainData, validateData, testData
end

function _loader:getDataByType(vector, targetType)
    -- Use cosine similarity to determine closest
    local closestId
    local closestSimilarity = -1
    local vectorNorm = torch.norm(vector)
    for id,embedding in pairs(self.embeddings) do
        local _,_, embeddingType = string.find(id, '^(%a+)')
        if embeddingType == targetType then
            local similarity = vector:dot(embedding)/(vectorNorm * torch.norm(embedding))
            if similarity >= closestSimilarity then
                closestId = id
                closestSimilarity = similarity
            end
        end
    end

    if not closestId then
        return
    end

    local _,_,id = string.find(closestId, '(%d+)')
    local data = self[targetType].data
    local value = data[id]
    local name = value and value.name

    if not name then
        -- must be by name
        id = tonumber(id)
        for k, v in pairs(data) do
            if v.id == id then
                name = k
                break
            end
        end
    end

    return name, closestSimilarity
end

function _loader:getChampionVector(champion)
    local championId
    for name, data in pairs(self.champion.data) do
        if name == champion then
            championId = data.id
            break
        end
    end

    assert(championId, 'Could not find champion: '..champion)
    return self.embeddings['champion'..championId]
end

function _loader:getLaneVector(lane)
    return self.embeddings[lane]
end

function _loader:getRoleVector(role)
    return self.embeddings[role]
end

function _loader:getVersionVector(version)
    return self.embeddings['version'..version]
end

function _loader:getRegionVector(region)
    return self.embeddings[region]
end

function _loader:getTierVector(tier)
    return self.embeddings[tier]
end

function _loader:sample(model, input, threshold)
    local targetTypes = {
        {count=dataset.maxSpellCount,type='spell'},
        {count=dataset.maxItemCount,type='item'},
        {count=dataset.maxRuneCount,type='rune'},
        {count=dataset.maxMasteryCount,type='mastery'},
    }

    local startIndex
    local endIndex = 0
    local prediction = model:forward(input)
    for _,targetType in ipairs(targetTypes) do
        startIndex = endIndex+1
        endIndex = startIndex+targetType.count-1

        local slice = prediction[{{startIndex,endIndex}}]
        for j=1,slice:size(1) do
            local name, confidence = self:getDataByType(slice[j], targetType.type)
            if name and confidence > threshold then
                print('datatype: '..targetType.type..', name: '..name..', confidence: '..confidence)
            end
        end
    end
end

function _data:__init(loader, inputs, targets, outcomes, batchsize, onlywins)
    self.loader = loader

    if onlywins then
        local j = 1
        local wins = torch.LongTensor(outcomes[torch.eq(outcomes,1)]:size(1))
        for i=1,outcomes:size(1) do
            if outcomes[i] == 1 then
                wins[j] = i
                j = j + 1
            end
        end
        inputs = inputs:index(1, wins)
        targets = targets:index(1, wins)
        outcomes = outcomes:index(1, wins)
    end

    self.inputs = inputs
    self.targets = targets
    self.outcomes = outcomes

    local inputSize = self.inputs:size():totable()
    inputSize[1] = batchsize

    local targetSize = self.targets:size():totable()
    targetSize[1] = batchsize

    local inputType = torch.factory(inputs:type())
    local targetType = torch.factory(targets:type())
    local outcomeType = torch.factory(outcomes:type())

    self.batch = {
        inputs=inputType():resize(unpack(inputSize)):zero(),
        targets=targetType():resize(unpack(targetSize)):zero(),
        outcomes=outcomeType():resize(batchsize):fill(1)
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
        self.batch.outcomes:index(self.outcomes, 1, batchIndices)

        return self.batch.inputs, self.batch.targets, self.batch.outcomes
    end
end

return dataset
