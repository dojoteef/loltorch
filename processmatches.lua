local cjson = require('cjson')
local dataset = require('dataset')
local file = require('pl.file')
local path = require('pl.path')
local tds = require('tds')
local threads = require('threads')
local torch = require('torch')
local utils = require('utils')
require('tds')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Get matches from the League of Legends API')
cmd:text()
cmd:text('Options')
cmd:option('-matchfile','matches.t7','the directory where the matches are located')
cmd:option('-datadir','dataset','the directory where to store the serialized dataset')
cmd:option('-threads',8,'the number of threads to use when processing the dataset')
cmd:option('-progress',100000,'display a progress update after x number of participants being processed')
cmd:option('-tensortype','torch.FloatTensor','what type of tensor to use')
cmd:text()

-- parse input opt
local opt = cmd:parse(arg)

local function clearTable(t)
    for i=1, #t do
        t[i] = nil
    end

    for _,v in pairs(t) do
        if type(v) == 'table' then
            clearTable(v)
        end
    end
end

local spells = {}
local function addSpells(embeddings, participant, target, offset)
    clearTable(spells)

    for i=1, dataset.maxSpellCount do
        local spellId = participant['spell'..i..'Id']
        if spellId then
            table.insert(spells, spellId)
        end
    end

    -- sort spells so they are in a consistent ordering
    table.sort(spells)

    for i=1,#spells do
        target[i+offset] = embeddings['s'..spells[i]]
    end

    return offset+dataset.maxSpellCount
end

local items = {}
local function addItems(embeddings, stats, target, offset)
    clearTable(items)

    for i=0, dataset.maxItemCount-1 do
        local itemId = stats['item'..i]
        if itemId and itemId > 0 then
            table.insert(items, itemId)
        end
    end

    -- sort items so they are in a consistent ordering
    table.sort(items)

    for i=1,#items do
        target[i+offset] = embeddings[dataset.itemId(items[i])]
    end

    return offset+dataset.maxItemCount
end

local runes = {blue={max=9},red={max=9},yellow={max=9},black={max=3}}
local function addRunes(embeddings, participant, target, offset, runeTypes)
    if participant.runes then
        clearTable(runes)

        for _,rune in ipairs(participant.runes) do
            local runeType = runeTypes[rune.runeId]
            table.insert(runes[runeType], rune)
        end

        local i = 1
        for _, runeType in ipairs(dataset.runeOrder) do
            -- sort runes so they are in a consistent ordering
            local runeList = runes[runeType]
            table.sort(runeList, function(r1, r2) return r1.runeId < r2.runeId end)

            local start = i
            for _,rune in ipairs(runeList) do
                for _=1, rune.rank do
                    target[i+offset] = embeddings[dataset.runeId(rune)]
                    i = i + 1
                end
            end

            i = start + runeList.max
        end
    end

    return offset+dataset.maxRuneCount
end

local masteries = {Ferocity={},Resolve={},Cunning={}}
local function addMasteries(embeddings, participant, target, offset, masteryTypes)
    if participant.masteries then
        clearTable(masteries)

        local i = 1
        for _,mastery in ipairs(participant.masteries) do
            local masteryType = masteryTypes[mastery.masteryId]
            table.insert(masteries[masteryType], mastery)
        end

        for _, masteryType in ipairs(dataset.masteryOrder) do
            -- sort masteries so they are in a consistent ordering
            local masteryList = masteries[masteryType]
            table.sort(masteryList, function(m1, m2) return m1.masteryId < m2.masteryId end)

            for _,mastery in ipairs(masteryList) do
                for _=1, mastery.rank do
                    target[i+offset] = embeddings[dataset.masteryId(mastery)]
                    i = i + 1
                end
            end
        end
    end

    return offset+dataset.maxMasteryCount
end

local function getRuneTypes(datadir)
    local runeFile = path.join(datadir, 'runes.json')
    local data = file.read(runeFile)
    local ok,runeInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..runeFile)
    end

    local runeTypes = {}
    for id,rune in pairs(runeInfo.data) do
        runeTypes[tonumber(id)] = rune.rune.type
    end

    return runeTypes
end

local function getMasteryTypes(datadir)
    local masteryFile = path.join(datadir, 'masteries.json')
    local data = file.read(masteryFile)
    local ok,masteryInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..masteryFile)
    end

    local masteryTypes = {}
    for tree, masteryList in pairs(masteryInfo.tree) do
        for _, masteryGroup in ipairs(masteryList) do
            for _,mastery in ipairs(masteryGroup) do
                if mastery ~= cjson.null then
                    masteryTypes[tonumber(mastery.masteryId)] = tree
                end
            end
        end
    end

    return masteryTypes
end

local function validParticipant(participant)
    return participant.stats ~= nil
end

local function countParticipants(matchlist)
    threads.serialization('threads.sharedserialize')
    local pool = threads.Threads(
        opt.threads,
        function()
            _G.cjson = require('cjson')
            _G.file = require('pl.file')
            _G.tds = require('tds')
        end
    )

    local participantsProcessed = tds.AtomicCounter()
    local participantsPerThread = torch.LongTensor(opt.threads)
    for i=1,opt.threads do
        pool:addjob(
            function(first,last)
                local threadid = _G.__threadid

                local participantCount = 0
                for j=first,last do
                    local data = _G.file.read(matchlist[j])
                    local ok,match = pcall(function() return _G.cjson.decode(data) end)
                    if ok then
                        for _,participant in ipairs(match.participants) do
                            if validParticipant(participant) then
                                participantCount = participantCount + 1
                            end

                            local processed = participantsProcessed:inc() + 1
                            if processed % opt.progress == 0 then
                                print('Processed '..processed..' participants')
                            end
                        end
                    end
                end
                
                participantsPerThread[threadid] = participantCount
            end,
            nil,
            utils.sliceIndices(i, opt.threads, matchlist)
        )
    end
    pool:synchronize()
    pool:terminate()

    return participantsPerThread
end

local function processMatches(matchlist, participantsPerThread, embeddings, datadir)
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
        opt.threads,
        function()
            _G.cjson = require('cjson')
            _G.file = require('pl.file')
            _G.path = require('pl.path')
            _G.tds = require('tds')
        end
    )

    local runeTypes = getRuneTypes(datadir)
    local masteryTypes = getMasteryTypes(datadir)
    local participantsProcessed = tds.AtomicCounter()
    for i=1,opt.threads do
        pool:addjob(
            function(first,last)
                local threadid = _G.__threadid

                local dataOffset = 0
                for t=1,threadid-1 do
                    dataOffset = dataOffset + participantsPerThread[t]
                end

                local dataIndex = dataOffset + 1
                for j=first,last do
                    local data = _G.file.read(matchlist[j])
                    local ok,match = pcall(function() return _G.cjson.decode(data) end)
                    if ok then
                        for _,participant in ipairs(match.participants) do
                            if validParticipant(participant) then
                                local input = inputs[dataIndex]
                                local target = targets[dataIndex]

                                local lane = dataset.normalizeLane(participant.timeline.lane)
                                local role = dataset.normalizeRole(lane, participant.timeline.role)

                                outcomes[dataIndex] = participant.stats.winner and 1 or -1

                                input[1] = embeddings[lane]
                                input[2] = embeddings[role]
                                input[3] = embeddings[dataset.championId(participant)]
                                input[4] = embeddings[dataset.versionFromString(match.matchVersion)]
                                input[5] = embeddings[string.lower(match.region)]
                                input[6] = embeddings[participant.highestAchievedSeasonTier or 'UNRANKED']

                                local offset = 0
                                offset = addSpells(embeddings, participant, target, offset)
                                offset = addItems(embeddings, participant.stats, target, offset)
                                offset = addRunes(embeddings, participant, target, offset, runeTypes)
                                addMasteries(embeddings, participant, target, offset, masteryTypes)

                                dataIndex = dataIndex + 1
                            end

                            local processed = participantsProcessed:inc() + 1
                            if processed % opt.progress == 0 then
                                print('Processed '..processed..' participants')
                            end
                        end
                    end
                end
            end,
            nil,
            utils.sliceIndices(i, opt.threads, matchlist)
        )
    end
    pool:synchronize()
    pool:terminate()

    return inputs, targets, outcomes
end

local function matchesToTensor()
    local timer = torch.Timer()

    torch.setdefaulttensortype(opt.tensortype)

    print('Count participants...')
    local matchlist = torch.load(opt.matchfile)
    local participantsPerThread =  countParticipants(matchlist)

    print('Process matches...')
    local embeddings = torch.load(path.join(opt.datadir, 'embeddings.t7'))
    local inputs, targets, outcomes = processMatches(matchlist, participantsPerThread, embeddings, opt.datadir)

    print('Saving tensors...')
    torch.save(path.join(opt.datadir,'inputs.t7'), inputs)
    torch.save(path.join(opt.datadir,'targets.t7'), targets)
    torch.save(path.join(opt.datadir,'outcomes.t7'), outcomes)

    print(string.format('Done in %s', utils.formatTime(timer)))
end

local function errorHandler(errmsg)
    errmsg = errmsg..'\n'..debug.traceback()
    print(errmsg)
end

local ok, errmsg = xpcall(matchesToTensor, errorHandler)
if not ok then
    print('Failed to process matches!')

    print(errmsg)
    os.exit()
end
