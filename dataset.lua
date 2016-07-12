local cjson = require('cjson')
local file = require('pl.file')
local path = require('pl.path')
local torch = require('torch')

local dataset = {
    maxSpellCount=2,
    maxItemCount = 7,
    maxBlueRuneCount = 9,
    maxRedRuneCount = 9,
    maxYellowRuneCount = 9,
    maxBlackRuneCount = 3,
    maxRuneCount = 30,
    maxMasteryCount = 30,
    runeOrder = {'blue','red','yellow','black'},
    masteryOrder = {'Cunning','Ferocity','Resolve'},
    validQueueTypes = {
        RANKED_SOLO_5x5 = true,
        RANKED_TEAM_5x5 = true,
        NORMAL_5x5_BLIND = true,
        NORMAL_5x5_DRAFT = true,
        GROUP_FINDER_5x5 = true,
        TEAM_BUILDER_DRAFT_RANKED_5x5 = true,
        TEAM_BUILDER_DRAFT_UNRANKED_5x5 = true,
    }
}

-- Using the knowledge from this forum post for
-- determining version, i.e. just using major/minor
-- https://developer.riotgames.com/discussion/community-discussion/show/FJT1rYsF
function dataset.parseVersion(versionString)
    local _,_,major,minor = string.find(versionString,"^(%d+)%.(%d+)")
    return major and minor and {major=tonumber(major), minor=tonumber(minor)}
end

function dataset.compareVersion(v1, v2)
    return v1.major < v2.major or (v1.major == v2.major and v1.minor < v2.minor)
end

function dataset.versionFromString(versionString)
    local version = dataset.parseVersion(versionString)
    return version and 'version'..version.major..'.'..version.minor
end

function dataset.championId(participant)
    return 'c'..participant.championId
end

function dataset.spellId(participant, index)
    return 's'..participant['spell'..index..'Id']
end

function dataset.runeId(rune)
    return 'r'..rune.runeId
end

function dataset.masteryId(mastery)
    return 'm'..mastery.masteryId
end

function dataset.itemId(itemId)
    return 'i'..itemId
end

-- Normalize on one representation for lane
function dataset.normalizeLane(lane)
    if lane == 'MIDDLE' then
        return 'MID'
    elseif lane == 'BOTTOM' then
        return 'BOT'
    else
        return lane
    end
end

-- Normalize role by fixing up 'NONE'
-- The function is FAR from perfect, but should help clean up some of the data.
-- Ideally it would also separate DUO into DUO_CARRY and DUO_SUPPORT, but that
-- seems like a difficult task.
function dataset.normalizeRole(lane, role)
    if role == 'NONE' then
        lane = dataset.normalizeLane(lane)
        if lane == 'BOT' then
            return 'DUO'
        else
            return 'SOLO'
        end
    else
        return role
    end
end

local targetCount = 0
for _,count in pairs(dataset) do
    if type(count) == 'number' then
        targetCount = targetCount + count
    end
end
dataset.targetCount = targetCount

local _data = torch.class('Dataset.Data', dataset)
local _loader = torch.class('Dataset.Loader', dataset)
local _sampler = torch.class('Dataset.Sampler', dataset)

function _loader:__init(datadir, testRatio, kfolds, batchsize)
    assert(path.isdir(datadir), 'Invalid data directory specified')
    self.embeddings = torch.load(path.join(datadir, 'embeddings.t7'))

    self.kfolds = kfolds
    self.testRatio = testRatio

    local inputs = torch.load(path.join(datadir, 'inputs.t7'))
    local targets = torch.load(path.join(datadir, 'targets.t7'))
    local outcomes = torch.load(path.join(datadir, 'outcomes.t7'))
    assert(inputs:type() == targets:type(), 'Dataset mismatch: input type ('..inputs:type()..'), target type ('..targets:type()..')')
    assert(inputs:size(1) == targets:size(1), 'Dataset mismatch: input size ('..inputs:size(1)..'), target size ('..targets:size(1)..')')
    assert(inputs:size(1) == outcomes:size(1), 'Dataset mismatch: input size ('..inputs:size(1)..'), outcome size ('..outcomes:size(1)..')')

    self.inputs = inputs
    self.targets = targets
    self.outcomes = outcomes
    self.batchSize = batchsize

    -- Make sure new tensors we create a compatible with the tensor we expect
    torch.setdefaulttensortype(torch.type(self.inputs))

    -- Split into training/test data
    local count = self.inputs:size(1)
    local trainIndex = count * (1 - self.testRatio)

    -- make training data equally divisible by kfolds * batchsize
    trainIndex = trainIndex - math.fmod(trainIndex, (self.kfolds*self.batchSize))

    local indices = torch.randperm(self.inputs:size(1)):long()
    self.folds = indices[{{1, trainIndex}}]:view(self.kfolds, -1)
    self.testData = dataset.Data(self.inputs, self.targets, self.outcomes, indices[{{trainIndex+1, count}}], self.batchSize)
end

function _loader:embeddingSize()
    local _,embedding = next(self.embeddings)
    return embedding:size(1)
end

function _loader:tensorType()
    return torch.type(self.inputs)
end

function _loader:getFold(k)
    local folds = torch.range(1, self.kfolds):long()

    local mask = torch.ne(folds, k):view(-1, 1):expandAs(self.folds)
    local trainIndices = self.folds:maskedSelect(mask)
    local trainData = dataset.Data(self.inputs, self.targets, self.outcomes, trainIndices:view(-1), self.batchSize)

    mask = torch.eq(folds, k):view(-1, 1):expandAs(self.folds)
    local validateIndices = self.folds:maskedSelect(mask)
    local validateData = dataset.Data(self.inputs, self.targets, self.outcomes, validateIndices:view(-1), self.batchSize, true)

    return trainData, validateData
end

function _sampler:__init(datadir)
    assert(path.isdir(datadir), 'Invalid data directory specified')
    self.embeddings = torch.load(path.join(datadir, 'embeddings.t7'))

    local data = file.read(path.join(datadir, 'champions.json'))
    local ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load champions.json')
    self.champion = json

    data = file.read(path.join(datadir, 'items.json'))
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load items.json')
    self.item = json

    data = file.read(path.join(datadir, 'masteries.json'))
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load masteries.json')
    self.mastery = json

    data = file.read(path.join(datadir, 'runes.json'))
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load runes.json')
    self.rune = json

    data = file.read(path.join(datadir, 'spells.json'))
    ok,json = pcall(function() return cjson.decode(data) end)
    assert(ok, 'Unable to load spells.json')
    self.spell = json
end

function _sampler:embeddingSize()
    local _,embedding = next(self.embeddings)
    return embedding:size(1)
end

function _sampler:getClosestData(vector, targetType, filter, list)
    -- Use cosine similarity to determine closest
    local closestId
    local closestSimilarity = -1
    local vectorNorm = vector:norm()
    for id,embedding in pairs(self.embeddings) do
        local _,_,idString = string.find(id, '(%d+)')
        local _,_,embeddingType = string.find(id, '^(%a+)')
        if embeddingType == string.sub(targetType, 1, 1) and (not filter or not filter(self, idString, list)) then
            local similarity = vector:dot(embedding)/(vectorNorm * embedding:norm())
            if similarity >= closestSimilarity then
                closestId = idString
                closestSimilarity = similarity
            end
        end
    end

    if not closestId then
        return
    end

    return self:getData(closestId, targetType), closestSimilarity
end

function _sampler:getData(idString, targetType)
    local data = self[targetType].data
    local value = data[idString]

    if not value then
        -- must be by name
        local id = tonumber(idString)
        for _, v in pairs(data) do
            if v.id == id or tonumber(v.key) == id then
                value = v
                break
            end
        end
    end

    return value
end

function _sampler:getChampionVector(champion)
    local championId
    for name, data in pairs(self.champion.data) do
        if name == champion then
            championId = data.key
            break
        end
    end

    assert(championId, 'Could not find champion: '..champion)
    return self.embeddings['c'..championId]
end

function _sampler:getLaneVector(lane)
    return self.embeddings[dataset.normalizeLane(lane)]
end

function _sampler:getRoleVector(lane, role)
    return self.embeddings[dataset.normalizeRole(dataset.normalizeLane(lane), role)]
end

function _sampler:getVersionVector(version)
    return self.embeddings[dataset.versionFromString(version)]
end

function _sampler:getRegionVector(region)
    return self.embeddings[region]
end

function _sampler:getTierVector(tier)
    return self.embeddings[tier]
end

function _sampler:filterSpells(spellId, list)
    local spell = self:getData(spellId, 'spell')
    for _, data in ipairs(list) do
        if data.datatype == 'spell' and data.data.id == spell.id then
            return true
        end
    end

    return false
end

function _sampler.getRuneFilter(runeType)
    return function(sampler, runeId, list)
        local rune = sampler:getData(runeId, 'rune')
        if rune.rune.type ~= runeType then
            return true
        end

        local runeCount = 0
        for _, data in ipairs(list) do
            if data.datatype == 'rune' and data.data.rune.type == rune.rune.type then
                runeCount = runeCount + 1
            end
        end

        if rune.rune.type == 'black' and runeCount >= 3 then
            return true
        elseif runeCount >= 9 then
            return true
        end

        return false
    end
end

local function obeysMasteryRules(mastery, masteries, tree)
    local ranks = {}
    for _, m in ipairs(masteries) do
        local rank = ranks[m.id] or {rank=0,max=m.ranks}
        rank.rank = rank.rank + 1

        ranks[m.id] = rank
    end

    if ranks[mastery.id] and ranks[mastery.id].rank >= mastery.ranks then
        return false
    end

    for _, level in ipairs(tree) do
        local levelMasteries = {}
        for _, treeItem in ipairs(level) do
            if treeItem ~= cjson.null then
                levelMasteries[tonumber(treeItem.masteryId)] = true
            end
        end

        if levelMasteries[mastery.id] then
            return true
        end

        local total = 0
        local requiredRank
        for masteryId in pairs(levelMasteries) do
            local rank = ranks[masteryId]
            if rank then
                total = total + rank.rank
                requiredRank = requiredRank or rank.max
            end
        end

        if not requiredRank or total < requiredRank then
            return false
        end
    end

    return true
end

local function getMasteryType(masteryId, masteryTree)
    for masteryType, masteryList in pairs(masteryTree) do
        for _, masteryGroup in ipairs(masteryList) do
            for _, mastery in ipairs(masteryGroup) do
                if mastery ~= cjson.null and tonumber(mastery.masteryId) == masteryId then
                    return masteryType
                end
            end
        end
    end
end

local function getMasteryIndex(mastery, masteryTree)
    for i, masteryType in ipairs(dataset.masteryOrder) do
        if masteryType == getMasteryType(mastery.id, masteryTree) then
            return i
        end
    end
end

local function getMasteryLevel(mastery, masteryTree)
    for i, level in ipairs(masteryTree) do
        for _, treeItem in ipairs(level) do
            if treeItem ~= cjson.null and tonumber(treeItem.masteryId) == mastery.id then
                return i
            end
        end
    end
end

function _sampler:filterMasteries(masteryId, list)
    local masteries = {}
    local mastery = self:getData(masteryId, 'mastery')
    local masteryIndex = getMasteryIndex(mastery, self.mastery.tree)
    for _, data in ipairs(list) do
        if data.datatype == 'mastery' then
            local otherMastery = data.data
            local otherMasteryType = getMasteryType(otherMastery.id, self.mastery.tree)
            local masteryTree = masteries[otherMasteryType] or {}
            table.insert(masteryTree, otherMastery)
            table.insert(masteries, otherMastery)

            masteries[otherMasteryType] = masteryTree
        end
    end

    local masteryType = getMasteryType(mastery.id, self.mastery.tree)
    local masteryTree = self.mastery.tree[masteryType]
    if #masteries > 0 then
        local lastMastery = masteries[#masteries]
        local lastMasteryIndex = getMasteryIndex(lastMastery, self.mastery.tree)
        if masteryIndex < lastMasteryIndex then
            return true
        end

        if masteryIndex == lastMasteryIndex and getMasteryLevel(mastery, masteryTree) < getMasteryLevel(lastMastery, masteryTree) then
            return true
        end
    end

    if not obeysMasteryRules(mastery, masteries[masteryType] or {}, masteryTree) then
        return true
    end

    return false
end

function _sampler:sample(model, input)
    local targetTypes = {
        {count=dataset.maxSpellCount,type='spell',filter=self.filterSpells},
        {count=dataset.maxItemCount,type='item'},
        {count=dataset.maxBlueRuneCount,type='rune',filter=self.getRuneFilter('blue')},
        {count=dataset.maxRedRuneCount,type='rune',filter=self.getRuneFilter('red')},
        {count=dataset.maxYellowRuneCount,type='rune',filter=self.getRuneFilter('yellow')},
        {count=dataset.maxBlackRuneCount,type='rune',filter=self.getRuneFilter('black')},
        {count=dataset.maxMasteryCount,type='mastery',filter=self.filterMasteries},
    }

    local output = {}
    local startIndex
    local endIndex = 0

    local prediction = model:forward(input):squeeze()
    for _,targetType in ipairs(targetTypes) do
        startIndex = endIndex+1
        endIndex = startIndex+targetType.count-1

        local slice = prediction[{{startIndex,endIndex}}]
        for j=1,slice:size(1) do
            local data, confidence = self:getClosestData(slice[j], targetType.type, targetType.filter, output)
            if data then
                table.insert(output, {data=data, datatype=targetType.type, confidence=confidence})
            end
        end
    end

    return output
end

function _data:__init(inputs, targets, outcomes, indices, batchsize)
    self.inputs = inputs
    self.targets = targets
    self.outcomes = outcomes
    self.indices = indices

    self.batch = {}
    self.batchSize = batchsize
end

function _data:tensorType()
    return torch.type(self.inputs)
end

function _data:size()
    return self.indices:nElement()
end

function _data:batchCount()
    return self:size()/self.batchSize
end

function _data:shuffle()
    if not self.startingIndices then
        self.startingIndices = self.indices:clone()
    end

    local perm = torch.randperm(self:size()):long()
    self.indices:indexCopy(1, perm, self.startingIndices)
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
