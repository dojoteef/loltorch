local cjson = require('cjson')
local dataset = require('dataset')
local dir = require('pl.dir')
local file = require('pl.file')
local lol = require('lol')
local path = require('pl.path')
local threads = require('threads')
local tablex = require('pl.tablex')
local tds = require('tds')
local torch = require('torch')
local utils = require('utils')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate a corpus from League of Legends data')
cmd:text()
cmd:text('Options')
cmd:option('-matchdir','matches','the directory where the matches are located')
cmd:option('-datadir','dataset','the directory where to store the serialized dataset')
cmd:option('-threads',8,'the number of threads to use when processing the dataset')
cmd:option('-progress',100000,'display a progress update after x number of matches being processed')
cmd:option('-outfile','matches.t7','the name of the output file which has the corpus')
cmd:option('-cutoff','6.7','ignore matches earlier than the cutoff')
cmd:text()

-- parse input opts
local opt = cmd:parse(arg)

local function getFeatures(datadir, filename, prefix, features)
    local featureFile = path.join(datadir, filename)
    local data = file.read(featureFile)
    local ok,info = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..featureFile)
    end

    for id,value in pairs(info.data) do
        if tonumber(id) then
            features[prefix..id] = true
        elseif tonumber(value.key) then
            features[prefix..value.key] = true
        end
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
        features['s'..spell.key] = true
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
        local versionString = dataset.versionFromString(version)
        if versionString then
            features[versionString] = true
        end
    end
end

local function getChampionFeatures(datadir, features)
    local championFile = path.join(datadir, 'champions.json')
    local data = file.read(championFile)
    local ok,championInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..championFile)
    end

    for _,champion in pairs(championInfo.data) do
        features['c'..champion.key] = true
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

    addFeatures({'SOLO', 'DUO', 'DUO_CARRY', 'DUO_SUPPORT'}, features) --roles
    addFeatures({'TOP', 'MID', 'BOT', 'JUNGLE'}, features) --lanes
    addFeatures(tablex.keys(lol.api.Regions), features) --regions
    addFeatures({'CHALLENGER', 'MASTER', 'DIAMOND', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'UNRANKED'}, features) --tiers

    return features
end

local function verifyFeature(entry, features)
    return features[entry] ~= nil
end

local function isMatchValid(match, features)
    if not match.queueType or not dataset.validQueueTypes[match.queueType] then return false end
    if dataset.compareVersion(dataset.parseVersion(match.matchVersion), dataset.parseVersion(opt.cutoff)) then return false end

    if not verifyFeature(string.lower(match.region), features) then return false end
    if not verifyFeature(dataset.versionFromString(match.matchVersion), features) then return false end

    for _,participant in ipairs(match.participants) do
        local lane = dataset.normalizeLane(participant.timeline.lane)
        if not verifyFeature(lane, features) then return false end

        local role = dataset.normalizeLane(lane, participant.timeline.role)
        if not verifyFeature(role, features) then return false end

        if not verifyFeature(dataset.championId(participant), features) then return false end
        if not verifyFeature(participant.highestAchievedSeasonTier, features) then return false end

        if participant.spell1Id and participant.spell1Id > 0 then
            if not verifyFeature(dataset.spellId(participant, 1), features) then return false end
        end

        if participant.spell2Id and participant.spell2Id > 0 then
            if not verifyFeature(dataset.spellId(participant, 2), features) then return false end
        end

        if participant.runes then
            for _,rune in ipairs(participant.runes) do
                if not verifyFeature(dataset.runeId(rune), features) then return false end
            end
        end

        if participant.masteries then
            for _,mastery in ipairs(participant.masteries) do
                if not verifyFeature(dataset.masteryId(mastery), features) then return false end
            end
        end

        if participant.stats then
            for i=0, dataset.maxItemCount-1 do
                local itemId = participant.stats['item'..i]
                if itemId and itemId > 0 then
                    if not verifyFeature(dataset.itemId(itemId), features) then return false end
                end
            end
        end
    end

    return true
end

local function verifyMatches(matchlist, features)
    threads.serialization('threads.sharedserialize')
    local pool = threads.Threads(
        opt.threads,
        function()
            _G.cjson = require('cjson')
            _G.file = require('pl.file')
            _G.tds = require('tds')
        end
    )

    local validMatchList = tds.Vec()
    local matchesProcessed = tds.AtomicCounter()
    for i=1,opt.threads do
        pool:addjob(
            function(first,last)
                local validMatches = _G.tds.Vec()
                for j=first,last do
                    local matchfile = matchlist[j]
                    local data = _G.file.read(matchfile)
                    local ok,match = pcall(function() return _G.cjson.decode(data) end)
                    if ok then
                        if isMatchValid(match, features) then
                            validMatches:insert(matchfile)
                        end

                        local processed = matchesProcessed:inc() + 1
                        if processed % opt.progress == 0 then
                            print('Analyzed '..processed..' matches')
                        end
                    end
                end

                return validMatches
            end,
            function(validMatches)
                for _,v in ipairs(validMatches) do
                    validMatchList:insert(v)
                end
            end,
            utils.sliceIndices(i, opt.threads, matchlist)
        )
    end
    pool:synchronize()
    pool:terminate()

    return validMatchList
end

local function validateMatches()
    local timer = torch.Timer()

    print('Verify matches...')
    local features = getAllFeatures(opt.datadir)
    local matchlist = tds.Vec(dir.getallfiles(opt.matchdir, '*.json'))
    local validMatches = verifyMatches(matchlist, features)
    print('#Valid matches: '..#validMatches)

    torch.save(path.join(opt.datadir, opt.outfile), validMatches)
    print(string.format('Done in %s', utils.formatTime(timer)))
end

local function errorHandler(errmsg)
    errmsg = errmsg..'\n'..debug.traceback()
    print(errmsg)
end

local ok, errmsg = xpcall(validateMatches, errorHandler)
if not ok then
    print('Failed to generate corpus!')

    print(errmsg)
    os.exit()
end
