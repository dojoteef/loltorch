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

function _loader:tensorType()
    return torch.type(self.inputs)
end

function _loader:load(batchsize, seed)
    local inputs = torch.load(self.dir..path.sep..'inputs.t7')
    local targets = torch.load(self.dir..path.sep..'targets.t7')
    local outcomes = torch.load(self.dir..path.sep..'outcomes.t7')
    assert(inputs:type() == targets:type(), 'Dataset mismatch: input type ('..inputs:type()..'), target type ('..targets:type()..')')
    assert(inputs:size(1) == targets:size(1), 'Dataset mismatch: input size ('..inputs:size(1)..'), target size ('..targets:size(1)..')')
    assert(inputs:size(1) == outcomes:size(1), 'Dataset mismatch: input size ('..inputs:size(1)..'), outcome size ('..outcomes:size(1)..')')

    -- Create a random number generator specifically for randomly permuting
    -- the data so that we can ensure we do it the same if continuing training.
    local generator = torch.Generator()
    if seed then
        torch.manualSeed(generator, seed)
    end

    -- Make sure new tensors we create a compatible with the tensor we expect
    torch.setdefaulttensortype(torch.type(inputs))

    local count = inputs:size(1)
    local trainIndex = math.floor(count * self.trainRatio)
    local validateIndex = trainIndex + math.floor(count * self.validateRatio)

    local indices = torch.randperm(inputs:size(1)):long()
    local view = indices[{{1, trainIndex}}]
    local trainData = dataset.Data(inputs, targets, outcomes, view, batchsize)

    view = indices[{{trainIndex+1, validateIndex}}]
    local validateData = dataset.Data(inputs, targets, outcomes, view, batchsize, true)

    view = indices[{{validateIndex+1, count}}]
    local testData = dataset.Data(inputs, targets, outcomes, view, batchsize, true)

    return {train=trainData, validate=validateData, test=testData, seed=torch.initialSeed(generator)}
end

function _loader:getDataByType(vector, targetType)
    -- Use cosine similarity to determine closest
    local closestId
    local closestSimilarity = -1
    local vectorNorm = vector:norm()
    for id,embedding in pairs(self.embeddings) do
        local _,_, embeddingType = string.find(id, '^(%a+)')
        if embeddingType == string.sub(targetType, 1, 1) then
            local similarity = vector:dot(embedding)/(vectorNorm * embedding:norm())
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
    return self.embeddings['c'..championId]
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

function _loader:sample(model, input)
    local targetTypes = {
        {count=dataset.maxSpellCount,type='spell'},
        {count=dataset.maxItemCount,type='item'},
        {count=dataset.maxRuneCount,type='rune'},
        {count=dataset.maxMasteryCount,type='mastery'},
    }

    local output = {}
    local startIndex
    local endIndex = 0

    local prediction = model:forward(input)
    for _,targetType in ipairs(targetTypes) do
        startIndex = endIndex+1
        endIndex = startIndex+targetType.count-1

        local slice = prediction[{{startIndex,endIndex}}]
        for j=1,slice:size(1) do
            local name, confidence = self:getDataByType(slice[j], targetType.type)
            if name then
                table.insert(output, {name=name, datatype=targetType.type, confidence=confidence})
            end
        end
    end

    return output
end

function _data:__init(inputs, targets, outcomes, indices, batchsize, onlywins)
    if onlywins then
        local winCount = 0
        for i=1,indices:size(1) do
            if outcomes[indices[i]] == 1 then
                winCount = winCount + 1
            end
        end

        local j = 0
        local wins = torch.LongTensor(winCount)
        for i=1,indices:size(1) do
            if outcomes[indices[i]] == 1 then
                j = j + 1
                wins[j] = i
            end
        end

        indices = wins
    end

    self.inputs = inputs
    self.targets = targets
    self.outcomes = outcomes
    self.indices = indices

    self.batch = {}
    self.batchSize = batchsize

    self:resetBatch()
end

function _data:tensorType()
    return torch.type(self.inputs)
end

function _data:size()
    return self.indices:size(1)
end

function _data:batchCount()
    return self:size()/self.batchSize
end

function _data:resetBatch()
    local perm = torch.randperm(self.indices:size(1)):long()
    self.indices = self.indices:indexCopy(1, perm, self.indices)
end

function _data:batchForId(id)
    local batch = self.batch[id]
    if not batch then
        local inputSize = self.inputs:size():totable()
        inputSize[1] = self.batchSize

        local targetSize = self.targets:size():totable()
        targetSize[1] = self.batchSize

        local inputType = torch.factory(self.inputs:type())
        local targetType = torch.factory(self.targets:type())
        local outcomeType = torch.factory(self.outcomes:type())

        batch = {
            inputs=inputType():resize(unpack(inputSize)):zero(),
            targets=targetType():resize(unpack(targetSize)):zero(),
            outcomes=outcomeType():resize(self.batchSize):fill(1)
        }

        self.batch[id] = batch
    end

    return batch
end

function _data:getBatch(i, id)
    if i <= self:batchCount() then
        local batchIndex = (i-1)*self.batchSize+1
        local endIndex = math.min(self.indices:size(1), batchIndex+self.batchSize-1)
        local batchIndices = self.indices[{{batchIndex, endIndex}}]

        local batch = self:batchForId(id or 1)
        batch.inputs:index(self.inputs, 1, batchIndices)
        batch.targets:index(self.targets, 1, batchIndices)
        batch.outcomes:index(self.outcomes, 1, batchIndices)

        return batch.inputs, batch.targets, batch.outcomes
    end
end

return dataset
