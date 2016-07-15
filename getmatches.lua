local cjson = require('cjson')
local dataset = require('dataset')
local file = require('pl.file')
local lol = require('lol')
local path = require('pl.path')
local stringx = require('pl.stringx')
local tds = require('tds')
local torch = require('torch')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Get matches from the League of Legends API')
cmd:text()
cmd:text('Options')
cmd:option('-cutoff','6.11','the lowest version to pull matches for')
cmd:option('-region','na','name of the region to get matches from')
cmd:option('-outdir','matches','the directory to store the matches in')
cmd:option('-tier','challenger','what tier matches to get, i.e. challenger, master, etc')
cmd:option('-verbose',false,'whether to have verbose output')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

local api = lol.api('.keys/devel', opt.region, '.cache', {
    maxRetries=3,
    rateLimits={
        {interval=10, count=10},
        {interval=600, count=500}
    },
    cacheSize={
        memory={size=math.huge,count=50}
    }
})

local league = lol.league(api)
local match = lol.match(api)
local matchlist = lol.matchlist(api)
local cutoff = dataset.parseVersion(opt.cutoff)

local ignoreFile = path.join(opt.outdir, opt.region, '.ignorelist')
local ignoreList = path.exists(ignoreFile) and torch.load(ignoreFile) or tds.Hash()

local function onMatchResponse(res, code, matchData)
    -- Status 200 means success and code being nil means we retrieved the data from the cache
    if code ~= nil and code ~= 200 then
        if code == 404 then
            -- while this isn't very efficient if there are a large number of missing matches that doesn't
            -- seem to be the case in practice so opting for a quick solution
            ignoreList[tostring(matchData.id)] = true
            torch.save(ignoreFile, ignoreList)
        end

        if opt.verbose then
            print('Received error ('..code..') trying to retreive match: '..stringx.shorten(cjson.encode(res), 100))
        end
        return
    end

    if not (res.matchId and res.region and res.matchVersion) then
        return
    end

    local version = dataset.parseVersion(res.matchVersion)
    local matchfile = path.join(opt.outdir, res.region, res.matchId..'.json')
    if not version or dataset.compareVersion(version, cutoff) then
        matchData.cutoff = true
    else
        file.write(matchfile, cjson.encode(res))
    end
end

local function onMatchResponseDecorator(matchData)
    return function (res, code)
        onMatchResponse(res, code, matchData)
    end
end

local function onMatchlistResponse(res, code)
    -- Status 200 means success and code being nil means we retrieved the data from the cache
    if code ~= nil and code ~= 200 then
        if opt.verbose then
            print('Received error ('..code..') trying to retreive matchlist: '..stringx.shorten(cjson.encode(res), 100))
        end
        return
    end

    -- Matches are received in descending chronological order (newest matches
    -- first), so keep getting matches until you run into a match with a
    -- version that is too old at which point move on to the next summoner.
    if res.matches then
        for _, entry in pairs(res.matches) do
            local matchData = {id=entry.matchId}
            if not ignoreList[tostring(entry.matchId)] then
                match:getById(entry.matchId, {callback=onMatchResponseDecorator(matchData)})
            end

            if matchData.cutoff then
                break
            end
        end
    end
end

local function onLeagueResponse(res, code)
    -- Status 200 means success and code being nil means we retrieved the data from the cache
    if code ~= nil and code ~= 200 then
        if opt.verbose then
            print('Received error ('..code..') trying to retreive league: '..stringx.shorten(cjson.encode(res), 100))
        end
        return
    end

    for _, entry in pairs(res.entries) do
        matchlist:getBySummonerId(entry.playerOrTeamId, {
            callback=onMatchlistResponse,
            filters={
                seasons={'SEASON2016'},
                rankedQueues={'RANKED_SOLO_5x5','TEAM_BUILDER_DRAFT_RANKED_5x5'}
            },
        })
    end
end

print('getting '..opt.region..' league')
league:getLeague(opt.tier, 'RANKED_SOLO_5x5', {callback=onLeagueResponse})
