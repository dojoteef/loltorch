local cjson = require('cjson')
local dataset = require('dataset')
local dir = require('pl.dir')
local file = require('pl.file')
local path = require('pl.path')
local torch = require('torch')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Get matches from the League of Legends API')
cmd:text()
cmd:text('Options')
cmd:option('-matchdir','matches','the directory where the matches are located')
cmd:option('-datadir','dataset','the directory where to store the serialized dataset')
cmd:option('-randomize',false,'whether to randomize or used sorted order for classification index')
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

local function matchesToTensor(matchdir, datadir)
    local timer = torch.Timer()

    print('timer: ', timer:time().real)
    print('verify data mapping...')

    local leagues = {}
    local invalidMatches = {}
    local participantCount = 0
    local embeddings = torch.load(datadir..path.sep..'embeddings.t7')
    for _,filename in pairs(dir.getallfiles(matchdir, '*.json')) do
        if string.find(filename, '-league.json') then
            local data = file.read(filename)
            local ok,league = pcall(function() return cjson.decode(data) end)
            if not ok then
                error('Unable to load: '..filename)
            end

            for playerId, leaguePoints in league do
                table.insert(leagues, {playerId=playerId, leaguePoints=leaguePoints})
            end
        else
            local data = file.read(filename)
            local ok,match = pcall(function() return cjson.decode(data) end)
            if ok then
                local valid = verifyEmbedding(string.lower(match.region), embeddings)
                valid = valid and verifyEmbedding(dataset.versionFromString(match.matchVersion), embeddings)

                for _,participant in ipairs(match.participants) do
                    participantCount = participantCount + 1

                    valid = valid and verifyEmbedding(getChampionId(participant), embeddings)
                    valid = valid and verifyEmbedding(getLane(participant.timeline.lane), embeddings)
                    valid = valid and verifyEmbedding(participant.timeline.role, embeddings)
                    valid = valid and verifyEmbedding(participant.highestAchievedSeasonTier, embeddings)

                    if participant.spell1Id and participant.spell1Id > 0 then
                        valid = valid and verifyEmbedding(getSpellId(participant, 1), embeddings)
                    end

                    if participant.spell2Id and participant.spell2Id > 0 then
                        valid = valid and verifyEmbedding(getSpellId(participant, 2), embeddings)
                    end

                    if participant.runes then
                        for _,rune in ipairs(participant.runes) do
                            valid = valid and verifyEmbedding(getRuneId(rune), embeddings)
                        end
                    end

                    if participant.masteries then
                        for _,mastery in ipairs(participant.masteries) do
                            valid = valid and verifyEmbedding(getMasteryId(mastery), embeddings)
                        end
                    end

                    if participant.stats then
                        for i=0, dataset.maxItemCount-1 do
                            local itemId = participant.stats['item'..i]
                            if itemId and itemId > 0 then
                                valid = valid and verifyEmbedding(getItemId(itemId), embeddings)
                            end
                        end
                    end
                end

                if not valid then
                    invalidMatches[filename] = true
                end
            end
        end
    end
    print('# participants: '..participantCount)

    local rankings = {}
    table.sort(leagues, function(l,r) return l.leaguePoints < r.leaguePoints and l.playerId < r.playerId end)
    for rank,player in ipairs(leagues) do
        rankings[player.playerId] = rank
    end

    print('timer: ', timer:time().real)
    print('loading matches...')
    local _,embedding = next(embeddings)
    local embeddingSize = embedding:size(1)

    local inputLen = 7
    local targetlen = dataset.maxSpellCount + dataset.maxItemCount + dataset.maxRuneCount + dataset.maxMasteryCount
    local targets = torch.FloatTensor(participantCount, targetlen, embeddingSize):zero()
    local inputs = torch.FloatTensor(participantCount, inputLen, embeddingSize):zero()

    local dataIndex = 1
    for _,matchFile in pairs(dir.getallfiles(matchdir, '*.json')) do
        if not invalidMatches[matchFile] then
            local data = file.read(matchFile)
            local ok,match = pcall(function() return cjson.decode(data) end)
            if ok then
                for _,participant in ipairs(match.participants) do
                    if validParticipant(participant) then
                        local input = inputs[dataIndex]
                        local target = targets[dataIndex]
                        if dataIndex % 100000 == 0 then
                            print('Processed '..dataIndex..' participants')
                        end

                        input[1] = embeddings[participant.timeline.role]
                        input[2] = embeddings[getChampionId(participant)]
                        input[3] = embeddings[getLane(participant.timeline.lane)]
                        input[4] = embeddings[participant.stats.winner and 'WIN' or 'LOSS']
                        input[5] = embeddings[dataset.versionFromString(match.matchVersion)]
                        input[6] = embeddings[string.lower(match.region)]
                        input[7] = embeddings[participant.highestAchievedSeasonTier or 'UNRANKED']

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


    print('saving tensors...')
    torch.save(datadir..path.sep..'einputs.t7', inputs)
    torch.save(datadir..path.sep..'etargets.t7', targets)

    print('done in time (seconds): ', timer:time().real)
end

matchesToTensor(params.matchdir, params.datadir)
