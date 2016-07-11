local dataset = require('dataset')
local path = require('pl.path')
local threads = require('threads')
local torch = require('torch')
local utils = require('utils')
require('tds')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate a corpus from League of Legends data')
cmd:text()
cmd:text('Options')
cmd:option('-matchfile','matches.t7','the directory where the matches are located')
cmd:option('-datadir','dataset','the directory where to store the serialized dataset')
cmd:option('-threads',8,'the number of threads to use when processing the dataset')
cmd:option('-progress',100000,'display a progress update after x number of participants being processed')
cmd:option('-outfile','corpus.txt','the name of the output file which has the corpus')
cmd:text()

-- parse input opts
local opt = cmd:parse(arg)

local function getSpells(participant, sentence)
    for i=1, dataset.maxSpellCount do
        local spellId = participant['spell'..i..'Id']
        if spellId then
            table.insert(sentence, dataset.spellId(participant, i))
        end
    end
end

local function getItems(stats, sentence)
    for i=0, dataset.maxItemCount-1 do
        local itemId = stats['item'..i]
        if itemId and itemId > 0 then
            table.insert(sentence, dataset.itemId(itemId))
        end
    end
end

local function getRunes(participant, sentence)
    if participant.runes then
        for _,rune in ipairs(participant.runes) do
            table.insert(sentence, dataset.runeId(rune))
        end
    end
end

local function getMasteries(participant, sentence)
    if participant.masteries then
        for _,mastery in ipairs(participant.masteries) do
            table.insert(sentence, dataset.masteryId(mastery))
        end
    end
end

local function getMatchData(match, participant)
    local input = {}
    local lane = dataset.normalizeLane(participant.timeline.lane)
    local role = dataset.normalizeRole(lane, participant.timeline.role)

    table.insert(input, lane)
    table.insert(input, role)
    table.insert(input, string.lower(match.region))
    table.insert(input, dataset.versionFromString(match.matchVersion))
    table.insert(input, dataset.championId(participant))
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

local function processMatches(matchlist, outfile, tmpfiles)
    threads.serialization('threads.sharedserialize')
    local pool = threads.Threads(
        opt.threads,
        function()
            _G.cjson = require('cjson')
            _G.file = require('pl.file')
        end
    )

    local fullCorpus = io.open(outfile, 'w+')
    for i=1,opt.threads do
        pool:addjob(
            function(first,last)
                local threadid = _G.__threadid

                local tmpfile = tmpfiles[threadid]
                local corpus = io.open(tmpfile, 'w+')
                local participantIndex = 0
                for j=first,last do
                    local data = _G.file.read(matchlist[j])
                    local ok,match = pcall(function() return _G.cjson.decode(data) end)
                    if ok then
                        for _,participant in ipairs(match.participants) do
                            if validParticipant(participant) then
                                local sentence = getMatchData(match, participant)
                                corpus:write(sentence)
                                corpus:write('\n')

                                participantIndex = participantIndex + 1
                                if participantIndex % opt.progress == 0 then
                                    print('Thread: '..threadid..' processed '..participantIndex..' participants')
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
            utils.sliceIndices(i, opt.threads, matchlist)
        )
    end
    pool:synchronize()
    pool:terminate()
end

local function matchesToCorpus(tmpfiles)
    local timer = torch.Timer()

    print('generate corpus...')
    local matchlist = torch.load(opt.matchfile)
    processMatches(matchlist, path.join(opt.datadir, opt.outfile), tmpfiles)

    print('done in time (seconds): ', timer:time().real)
end

local tmpfiles = {}
for _=1,opt.threads do
    table.insert(tmpfiles, os.tmpname())
end

local function errorHandler(errmsg)
    errmsg = errmsg..'\n'..debug.traceback()
    print(errmsg)

    for _,tmpfile in ipairs(tmpfiles) do
        os.remove(tmpfile)
    end
end

local ok, errmsg = xpcall(matchesToCorpus, errorHandler, tmpfiles)
if not ok then
    print('Failed to generate corpus!')

    print(errmsg)
    os.exit()
end
