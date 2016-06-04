local cjson = require('cjson')
local dataset = require('dataset')
local dir = require('pl.dir')
local file = require('pl.file')
local lol = require('lol')
local path = require('pl.path')
local threads = require('threads')
local tablex = require('pl.tablex')
local torch = require('torch')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate a corpus from League of Legends data')
cmd:text()
cmd:text('Options')
cmd:option('-matchdir','matches','the directory where the matches are located')
cmd:option('-datadir','dataset','the directory where to store the serialized dataset')
cmd:option('-threads',1,'the number of threads to use when processing the dataset')
cmd:option('-progress',100000,'display a progress update after x number of participants being processed')
cmd:option('-outfile','corpus.txt','the name of the output file which has the corpus')
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

local function getFeatures(datadir, filename, prefix, features)
    local featureFile = path.join(datadir, filename)
    local data = file.read(featureFile)
    local ok,info = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..featureFile)
    end

    for id in pairs(info.data) do
        features[prefix..id] = true
    end
end

local function getSpellFeatures(datadir, features)
    local spellFile = path.join(datadir, 'spells.json')
    local data = file.read(spellFile)
    local ok,spellInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..spellFile)
    end

    for _, spell in pairs(spellInfo.data) do
        features['s'..spell.id] = true
    end
end

local function getVersionFeatures(datadir, features)
    local versionFile = path.join(datadir, 'versions.json')
    local data = file.read(versionFile)
    local ok,versionInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..versionFile)
    end

    for _, version in ipairs(versionInfo) do
        features[dataset.versionFromString(version)] = true
    end
end

local function getChampionFeatures(datadir, features)
    local championFile = path.join(datadir, 'champions.json')
    local data = file.read(championFile)
    local ok,championInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..championFile)
    end

    for championId in pairs(championInfo.keys) do
        features['c'..championId] = true
    end
end

local function addFeatures(values, features)
    for _, value in ipairs(values) do
        features[value] = true
    end
end

local function getAllFeatures(datadir)
    local features = {}
    getSpellFeatures(datadir, features)
    getVersionFeatures(datadir, features)
    getChampionFeatures(datadir, features)
    getFeatures(datadir, 'runes.json', 'r', features)
    getFeatures(datadir, 'items.json', 'i', features)
    getFeatures(datadir, 'masteries.json', 'm', features)

    addFeatures({'DUO', 'NONE', 'SOLO', 'DUO_CARRY', 'DUO_SUPPORT'}, features) --roles
    addFeatures({'MID', 'TOP', 'JUNGLE', 'BOT'}, features) --lanes
    addFeatures(tablex.keys(lol.api.Regions), features) --regions
    addFeatures({'CHALLENGER', 'MASTER', 'DIAMOND', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'UNRANKED'}, features) --tiers

    return features
end

local function getChampionId(participant)
    return 'c'..participant.championId
end

local function getSpellId(participant, index)
    return 's'..participant['spell'..index..'Id']
end

local function getRuneId(rune)
    return 'r'..rune.runeId
end

local function getMasteryId(mastery)
    return 'm'..mastery.masteryId
end

local function getItemId(itemId)
    return 'i'..itemId
end

local function getSpells(participant, sentence)
    for i=1, dataset.maxSpellCount do
        local spellId = participant['spell'..i..'Id']
        if spellId then
            table.insert(sentence, getSpellId(participant, i))
        end
    end
end

local function getItems(stats, sentence)
    for i=0, dataset.maxItemCount-1 do
        local itemId = stats['item'..i]
        if itemId and itemId > 0 then
            table.insert(sentence, getItemId(itemId))
        end
    end
end

local function getRunes(participant, sentence)
    if participant.runes then
        for _,rune in ipairs(participant.runes) do
            table.insert(sentence, getRuneId(rune))
        end
    end
end

local function getMasteries(participant, sentence)
    if participant.masteries then
        for _,mastery in ipairs(participant.masteries) do
            table.insert(sentence, getMasteryId(mastery))
        end
    end
end

local function getMatchData(match, participant)
    local input = {}
    table.insert(input, string.lower(match.region))
    table.insert(input, dataset.versionFromString(match.matchVersion))
    table.insert(input, getChampionId(participant))
    table.insert(input, getLane(participant.timeline.lane))
    table.insert(input, participant.timeline.role)
    table.insert(input, participant.highestAchievedSeasonTier)

    local target = {}
    getSpells(participant, target)
    getItems(participant.stats, target)
    getRunes(participant, target)
    getMasteries(participant, target)

    return table.concat(input, ' ')..'\n'..table.concat(target, ' ')
end

local function validParticipant(participant)
    return participant.stats ~= nil
end

local function verifyFeature(entry, features)
    return features[entry] ~= nil
end

local function isMatchValid(match, features)
    if not verifyFeature(string.lower(match.region), features) then return false end
    if not verifyFeature(dataset.versionFromString(match.matchVersion), features) then return false end

    for _,participant in ipairs(match.participants) do
        if not verifyFeature(getChampionId(participant), features) then return false end
        if not verifyFeature(getLane(participant.timeline.lane), features) then return false end
        if not verifyFeature(participant.timeline.role, features) then return false end
        if not verifyFeature(participant.highestAchievedSeasonTier, features) then return false end

        if participant.spell1Id and participant.spell1Id > 0 then
            if not verifyFeature(getSpellId(participant, 1), features) then return false end
        end

        if participant.spell2Id and participant.spell2Id > 0 then
            if not verifyFeature(getSpellId(participant, 2), features) then return false end
        end

        if participant.runes then
            for _,rune in ipairs(participant.runes) do
                if not verifyFeature(getRuneId(rune), features) then return false end
            end
        end

        if participant.masteries then
            for _,mastery in ipairs(participant.masteries) do
                if not verifyFeature(getMasteryId(mastery), features) then return false end
            end
        end

        if participant.stats then
            for i=0, dataset.maxItemCount-1 do
                local itemId = participant.stats['item'..i]
                if itemId and itemId > 0 then
                    if not verifyFeature(getItemId(itemId), features) then return false end
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

local function verifyMatches(matchlist, features)
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
                        if isMatchValid(match, features) then
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

    return invalidMatchList, torch.sum(participantsPerThread)
end

local function processMatches(matchlist, invalidMatches, outfile, tmpfiles)
    threads.serialization('threads.sharedserialize')
    local pool = threads.Threads(
        params.threads,
        function()
            _G['cjson'] = require('cjson')
            _G['file'] = require('pl.file')
        end
    )

    local fullCorpus = io.open(outfile, 'w+')
    for i=1,params.threads do
        pool:addjob(
            function(matches)
                local threadid = _G['__threadid']

                local tmpfile = tmpfiles[threadid]
                local corpus = io.open(tmpfile, 'w+')
                local participantIndex = 0
                for _,matchFile in pairs(matches) do
                    if not invalidMatches[matchFile] then
                        local data = _G['file'].read(matchFile)
                        local ok,match = pcall(function() return _G['cjson'].decode(data) end)
                        if ok then
                            for _,participant in ipairs(match.participants) do
                                if validParticipant(participant) then
                                    local sentence = getMatchData(match, participant)
                                    corpus:write(sentence)
                                    corpus:write('\n')

                                    participantIndex = participantIndex + 1
                                    if participantIndex % params.progress == 0 then
                                        print('Thread: '..threadid..' processed '..participantIndex..' participants')
                                    end
                                end
                            end
                        end
                    end
                end

                corpus:flush()
                corpus:close()
                
                return tmpfile
            end,
            function(tmpfile)
                local corpus = io.open(tmpfile, 'r')
                for line in corpus:lines() do
                    fullCorpus:write(line)
                    fullCorpus:write('\n')
                end

                fullCorpus:flush()
                corpus:close()
            end,
            matchesForThread(i, matchlist)
        )
    end
    pool:synchronize()
    pool:terminate()
end

local function matchesToCorpus(matchdir, datadir, outfile, tmpfiles)
    local timer = torch.Timer()

    print('timer: ', timer:time().real)
    print('verify matches...')
    local features = getAllFeatures(datadir)
    local matchlist = dir.getallfiles(matchdir, '*.json')
    local invalidMatches, participantCount = verifyMatches(matchlist, features)

    print('# participants: '..participantCount, '# invalid matches: '..tablex.size(invalidMatches))

    print('timer: ', timer:time().real)
    print('process matches...')
    processMatches(matchlist, invalidMatches, path.join(datadir, outfile), tmpfiles)

    print('done in time (seconds): ', timer:time().real)
end

local tmpfiles = {}
for _=1,params.threads do
    table.insert(tmpfiles, os.tmpname())
end

local function errorHandler(errmsg)
    errmsg = errmsg..'\n'..debug.traceback()
    print(errmsg)

    for _,tmpfile in ipairs(tmpfiles) do
        os.remove(tmpfile)
    end
end

local ok, errmsg = xpcall(matchesToCorpus, errorHandler, params.matchdir, params.datadir, params.outfile, tmpfiles)
if not ok then
    print('Failed to generate corpus!')

    print(errmsg)
    os.exit()
end
