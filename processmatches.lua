local dataset = require('dataset')
local dir = require('pl.dir')
local path = require('pl.path')
local threads = require('threads')
local tablex = require('pl.tablex')
local torch = require('torch')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Get matches from the League of Legends API')
cmd:text()
cmd:text('Options')
cmd:option('-matchdir','matches','the directory where the matches are located')
cmd:option('-datadir','dataset','the directory where to store the serialized dataset')
cmd:option('-threads',1,'the number of threads to use when processing the dataset')
cmd:option('-progress',100000,'display a progress update after x number of participants being processed')
cmd:option('-tensortype','torch.FloatTensor','what type of tensor to use')
cmd:text()

-- parse input params
local params = cmd:parse(arg)

-- Normalize on one representation for lane
local function getLane(lane)
    if lane == 'MIDDLE' then
        return 'MID'
    elseif lane == 'BOTTOM' then
        return 'BOT'
    else
        return lane
    end
end

local function getChampionId(participant)
    return 'champion'..participant.championId
end

local function getSpellId(participant, index)
    return 'spell'..participant['spell'..index..'Id']
end

local function getRuneId(rune)
    return 'rune'..rune.runeId
end

local function getMasteryId(mastery)
    return 'mastery'..mastery.masteryId
end

local function getItemId(itemId)
    return 'item'..itemId
end

local function addSpells(embeddings, participant, target, offset)
    for i=1, dataset.maxSpellCount do
        local spellId = participant['spell'..i..'Id']
        if spellId then
            target[i+offset] = embeddings[getSpellId(participant, i)]
        end
    end

    return offset+dataset.maxSpellCount
end

local function addItems(embeddings, stats, target, offset)
    for i=0, dataset.maxItemCount-1 do
        local itemId = stats['item'..i]
        if itemId and itemId > 0 then
            target[i+offset] = embeddings[getItemId(itemId)]
        end
    end

    return offset+dataset.maxItemCount
end

local function addRunes(embeddings, participant, target, offset)
    if participant.runes then
        local i = 1
        for _,rune in ipairs(participant.runes) do
            for _=1, rune.rank do
                target[i+offset] = embeddings[getRuneId(rune)]
                i = i + 1
            end
        end
    end

    return offset+dataset.maxRuneCount
end

local function addMasteries(embeddings, participant, target, offset)
    if participant.masteries then
        local i = 1
        for _,mastery in ipairs(participant.masteries) do
            for _=1, mastery.rank do
                target[i+offset] = embeddings[getMasteryId(mastery)]
                i = i + 1
            end
        end
    end

    return offset+dataset.maxMasteryCount
end

local function validParticipant(participant)
    return participant.stats ~= nil
end

local function verifyEmbedding(entry, embeddings)
    return embeddings[entry] ~= nil
end

local function isMatchValid(match, embeddings)
    if not verifyEmbedding(string.lower(match.region), embeddings) then return false end
    if not verifyEmbedding(dataset.versionFromString(match.matchVersion), embeddings) then return false end

    for _,participant in ipairs(match.participants) do
        if not verifyEmbedding(getChampionId(participant), embeddings) then return false end
        if not verifyEmbedding(getLane(participant.timeline.lane), embeddings) then return false end
        if not verifyEmbedding(participant.timeline.role, embeddings) then return false end
        if not verifyEmbedding(participant.highestAchievedSeasonTier, embeddings) then return false end

        if participant.spell1Id and participant.spell1Id > 0 then
            if not verifyEmbedding(getSpellId(participant, 1), embeddings) then return false end
        end

        if participant.spell2Id and participant.spell2Id > 0 then
            if not verifyEmbedding(getSpellId(participant, 2), embeddings) then return false end
        end

        if participant.runes then
            for _,rune in ipairs(participant.runes) do
                if not verifyEmbedding(getRuneId(rune), embeddings) then return false end
            end
        end

        if participant.masteries then
            for _,mastery in ipairs(participant.masteries) do
                if not verifyEmbedding(getMasteryId(mastery), embeddings) then return false end
            end
        end

        if participant.stats then
            for i=0, dataset.maxItemCount-1 do
                local itemId = participant.stats['item'..i]
                if itemId and itemId > 0 then
                    if not verifyEmbedding(getItemId(itemId), embeddings) then return false end
                end
            end
        end
    end

    return true
end

local function matchesForThread(threadid, matchlist)
    local matchesPerThread = math.ceil(#matchlist/params.threads)
    return tablex.sub(matchlist, ((threadid-1)*matchesPerThread)+1, threadid*matchesPerThread)
end

local function verifyMatches(matchlist, embeddings)
    threads.serialization('threads.sharedserialize')
    local pool = threads.Threads(
        params.threads,
        function()
            _G['cjson'] = require('cjson')
            _G['file'] = require('pl.file')
            _G['tablex'] = require('pl.tablex')
        end
    )

    local invalidMatchList = {}
    local participantsPerThread = torch.LongTensor(params.threads)
    for i=1,params.threads do
        pool:addjob(
            function(matches)
                local participants = 0
                local invalidMatches = {}
                local threadid = _G['__threadid']
                for _,matchfile in ipairs(matches) do
                    local data = _G['file'].read(matchfile)
                    local ok,match = pcall(function() return _G['cjson'].decode(data) end)
                    if ok then
                        if isMatchValid(match, embeddings) then
                            local lastProgress = math.floor(participants / params.progress)
                            participants = participants + #match.participants
                            if lastProgress ~= math.floor(participants / params.progress) then
                                print('Thread: '..threadid..' verified '..participants..' participants')
                            end
                        else
                            invalidMatches[matchfile] = true
                        end
                    end
                end

                participantsPerThread[threadid] = participants
                return invalidMatches
            end,
            function(invalidMatches)
                invalidMatchList = _G['tablex'].merge(invalidMatchList, invalidMatches, true)
            end,
            matchesForThread(i, matchlist)
        )
    end
    pool:synchronize()
    pool:terminate()

    return invalidMatchList, participantsPerThread
end

local function processMatches(matchlist, invalidMatches, participantsPerThread, embeddings)
    local _,embedding = next(embeddings)
    local embeddingSize = embedding:size(1)

    local inputLen = 6
    local targetlen = dataset.maxSpellCount + dataset.maxItemCount + dataset.maxRuneCount + dataset.maxMasteryCount

    local participantCount = torch.sum(participantsPerThread)
    local inputs = torch.Tensor(participantCount, inputLen, embeddingSize):zero()
    local targets = torch.Tensor(participantCount, targetlen, embeddingSize):zero()
    local outcomes = torch.Tensor(participantCount)

    threads.serialization('threads.sharedserialize')
    local pool = threads.Threads(
        params.threads,
        function()
            _G['cjson'] = require('cjson')
            _G['file'] = require('pl.file')
        end
    )

    for i=1,params.threads do
        pool:addjob(
            function(matches)
                local threadid = _G['__threadid']

                local dataOffset = 0
                for t=1,threadid-1 do
                    dataOffset = dataOffset + participantsPerThread[t]
                end

                local dataIndex = dataOffset + 1
                for _,matchFile in pairs(matches) do
                    if not invalidMatches[matchFile] then
                        local data = _G['file'].read(matchFile)
                        local ok,match = pcall(function() return _G['cjson'].decode(data) end)
                        if ok then
                            for _,participant in ipairs(match.participants) do
                                if validParticipant(participant) then
                                    local input = inputs[dataIndex]
                                    local target = targets[dataIndex]

                                    local participantIndex = (dataIndex - dataOffset)
                                    if participantIndex % params.progress == 0 then
                                        print('Thread: '..threadid..' processed '..participantIndex..' participants')
                                    end

                                    outcomes[dataIndex] = participant.stats.winner and 1 or -1

                                    input[1] = embeddings[participant.timeline.role]
                                    input[2] = embeddings[getChampionId(participant)]
                                    input[3] = embeddings[getLane(participant.timeline.lane)]
                                    input[4] = embeddings[dataset.versionFromString(match.matchVersion)]
                                    input[5] = embeddings[string.lower(match.region)]
                                    input[6] = embeddings[participant.highestAchievedSeasonTier or 'UNRANKED']

                                    local offset = 0
                                    offset = addSpells(embeddings, participant, target, offset)
                                    offset = addItems(embeddings, participant.stats, target, offset)
                                    offset = addRunes(embeddings, participant, target, offset)
                                    addMasteries(embeddings, participant, target, offset)

                                    dataIndex = dataIndex + 1
                                end
                            end
                        end
                    end
                end
            end,
            nil,
            matchesForThread(i, matchlist)
        )
    end
    pool:synchronize()
    pool:terminate()

    return inputs, targets, outcomes
end

local function matchesToTensor(matchdir, datadir)
    local timer = torch.Timer()

    torch.setdefaulttensortype(params.tensortype)

    print('timer: ', timer:time().real)
    print('verify matches...')
    local embeddings = torch.load(datadir..path.sep..'embeddings.t7')
    local matchlist = dir.getallfiles(matchdir, '*.json')
    local invalidMatches, participantsPerThread = verifyMatches(matchlist, embeddings)

    print('# participants: '..torch.sum(participantsPerThread), '# invalid matches: '..tablex.size(invalidMatches))

    print('timer: ', timer:time().real)
    print('process matches...')
    local inputs, targets, outcomes = processMatches(matchlist, invalidMatches, participantsPerThread, embeddings)

    print('saving tensors...')
    torch.save(datadir..path.sep..'inputs.t7', inputs)
    torch.save(datadir..path.sep..'targets.t7', targets)
    torch.save(datadir..path.sep..'outcomes.t7', outcomes)

    print('done in time (seconds): ', timer:time().real)
end

matchesToTensor(params.matchdir, params.datadir)
