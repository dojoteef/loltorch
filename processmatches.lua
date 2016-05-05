local cjson = require('cjson')
local dataset = require('dataset')
local dir = require('pl.dir')
local file = require('pl.file')
local path = require('pl.path')
local torch = require('torch')
local tablex = require('pl.tablex')

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

local nullId = 1
local function addSpells(mapping, participant, target, tmp, offset)
    for i=1, dataset.maxSpellCount do
        local spellId = participant['spell'..i..'Id']
        table.insert(tmp, spellId and mapping.spells[spellId] or nullId)
    end
    table.sort(tmp)

    for i=1, dataset.maxSpellCount do
        target[i] = tmp[i]
        tmp[i] = nil
    end

    return offset+dataset.maxSpellCount
end

local function addItems(mapping, stats, target, tmp, offset)
    for i=0, dataset.maxItemCount-1 do
        local itemId = stats['item'..i]
        table.insert(tmp, itemId and mapping.items[itemId] or nullId)
    end

    while #tmp < dataset.maxItemCount do
        table.insert(tmp, nullId)
    end
    table.sort(tmp)

    for i=1, dataset.maxItemCount do
        target[i+offset] = tmp[i]
        tmp[i] = nil
    end

    return offset+dataset.maxItemCount
end

local function addRunes(mapping, participant, target, tmp, offset)
    if participant.runes then
        for _,rune in ipairs(participant.runes) do
            for _=1, rune.rank do
                table.insert(tmp, mapping.runes[rune.runeId])
            end
        end
    end

    while #tmp < dataset.maxRuneCount do
        table.insert(tmp, nullId)
    end
    table.sort(tmp)

    for i,id in ipairs(tmp) do
        target[i+offset] = id
        tmp[i] = nil
    end

    return offset+dataset.maxRuneCount
end

local function addMasteries(mapping, participant, target, tmp, offset)
    if participant.masteries then
        for _,mastery in ipairs(participant.masteries) do
            for _=1, mastery.rank do
                table.insert(tmp, mapping.masteries[mastery.masteryId])
            end
        end
    end

    while #tmp < dataset.maxMasteryCount do
        table.insert(tmp, nullId)
    end
    table.sort(tmp)

    for i,id in ipairs(tmp) do
        target[i+offset] = id
        tmp[i] = nil
    end

    return offset+dataset.maxMasteryCount
end

local function tensorType(byteLength)
    if byteLength == 1 then
        return torch.ByteTensor
    elseif byteLength == 2 then
        return torch.ShortTensor
    elseif byteLength <= 4 then
        return torch.IntTensor
    else
        error('One or more of the category lists exceeds the max data size')
    end
end

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

local function validParticipant(participant)
    return participant.stats ~= nil
end

local function matchesToTensor(matchdir, datadir)
    local timer = torch.Timer()

    print('timer: ', timer:time().real)
    print('create data mapping...')
    local unordered = {
        inputs={
            lanes={},
            roles={},
            champions={},
            versions={},
            outcomes={WIN=true,LOSS=true},
            regions={},
            tiers={}
        },
        targets={
            runes={},
            spells={},
            masteries={},
            items={}
        }
    }

    local leagues = {}
    local participantCount = 0
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
                unordered.inputs.regions[match.region] = true
                unordered.inputs.versions[match.matchVersion] = true

                for _,participant in ipairs(match.participants) do
                    participantCount = participantCount + 1

                    unordered.inputs.roles[participant.timeline.role] = true
                    unordered.inputs.champions[participant.championId] = true
                    unordered.inputs.lanes[getLane(participant.timeline.lane)] = true
                    if participant.highestAchievedSeasonTier then
                        unordered.inputs.tiers[participant.highestAchievedSeasonTier] = true
                    end

                    if participant.spell1Id and participant.spell1Id > 0 then
                        unordered.targets.spells[participant.spell1Id] = true
                    end

                    if participant.spell2Id and participant.spell2Id > 0 then
                        unordered.targets.spells[participant.spell2Id] = true
                    end

                    if participant.runes then
                        for _,rune in ipairs(participant.runes) do
                            unordered.targets.runes[rune.runeId] = true
                        end
                    end

                    if participant.masteries then
                        for _,mastery in ipairs(participant.masteries) do
                            unordered.targets.masteries[mastery.masteryId] = true
                        end
                    end

                    if participant.stats then
                        for i=0, dataset.maxItemCount-1 do
                            local itemId = participant.stats['item'..i]
                            if itemId and itemId > 0 then
                                unordered.targets.items[itemId] = true
                            end
                        end
                    end
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

    local ordered = {}
    local datasizes={}
    for datatype, tables in pairs(unordered) do
        ordered[datatype] = {}
        datasizes[datatype] = 1
        for category, list in pairs(tables) do
            local sorted = {}
            for id in pairs(list) do
                table.insert(sorted, id)
            end

            table.sort(sorted)
            print('category '..category..': '..#sorted)
            datasizes[datatype] = datasizes[datatype] + #sorted
            ordered[datatype][category] = sorted
        end
    end

    local inputLen = tablex.size(ordered.inputs)
    local targetlen = dataset.maxSpellCount + dataset.maxItemCount + dataset.maxRuneCount + dataset.maxMasteryCount

    local mapping = {}
    for datatype, tables in pairs(ordered) do
        local indices
        if params.randomize then
            indices = torch.randperm(datasizes[datatype])
        else
            indices = torch.range(1,datasizes[datatype])
        end

        -- leave space for null/empty which is defined to be 1
        local data = {size=1}
        for category, list in pairs(tables) do
            local map = {}
            for i,id in ipairs(list) do
                map[id] = indices[data.size + i]
            end

            data[category] = map
            data.size = data.size + #list
        end
        -- how many bytes are needed to encode the list
        data.byteLength = math.ceil(math.log(data.size, 2) / 8)

        mapping[datatype] = data
    end

    print('timer: ', timer:time().real)
    print('loading matches...')
    local targets = tensorType(mapping.targets.byteLength)(participantCount, targetlen)
    local inputs = tensorType(mapping.inputs.byteLength)(participantCount, inputLen)

    local tmp = {}
    local dataIndex = 1
    for _,matchFile in pairs(dir.getallfiles(matchdir, '*.json')) do
        local data = file.read(matchFile)
        local ok,match = pcall(function() return cjson.decode(data) end)
        if ok then
            for _,participant in ipairs(match.participants) do
                if validParticipant(participant) then
                    local input = inputs[dataIndex]
                    local target = targets[dataIndex]
                    dataIndex = dataIndex + 1

                    input[1] = mapping.inputs.roles[participant.timeline.role]
                    input[2] = mapping.inputs.champions[participant.championId]
                    input[3] = mapping.inputs.lanes[getLane(participant.timeline.lane)]
                    input[4] = mapping.inputs.outcomes[participant.stats.winner and 'WIN' or 'LOSS']
                    input[5] = mapping.inputs.versions[match.matchVersion]
                    input[6] = mapping.inputs.regions[match.region]
                    input[7] = mapping.inputs.tiers[participant.highestAchievedSeasonTier or 'UNRANKED']

                    local offset = 0
                    offset = addSpells(mapping.targets, participant, target, tmp, offset)
                    offset = addItems(mapping.targets, participant.stats, target, tmp, offset)
                    offset = addRunes(mapping.targets, participant, target, tmp, offset)
                    addMasteries(mapping.targets, participant, target, tmp, offset)
                end
            end
        end
    end


    print('saving tensors...')
    torch.save(datadir..path.sep..'map.t7', mapping)
    torch.save(datadir..path.sep..'inputs.t7', inputs)
    torch.save(datadir..path.sep..'targets.t7', targets)

    print('done in time (seconds): ', timer:time().real)
end

matchesToTensor(params.matchdir, params.datadir)
