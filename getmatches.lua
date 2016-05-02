local cjson = require('cjson')
local file = require('pl.file')
local lol = require('lol')
local path = require('pl.path')
local stringx = require('pl.stringx')
local torch = require('torch')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Get matches from the League of Legends API')
cmd:text()
cmd:text('Options')
cmd:option('-region','na','name of the region to get matches from')
cmd:option('-outdir','matches','the directory to store the matches in')
cmd:option('-tier','challenger','what tier matches to get, i.e. challenger, master, etc')
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

local function onMatchResponse(res, code)
    -- Status 200 means success and code being nil means we retrieved the data from the cache
    if code ~= nil and code ~= 200 then
        print('Received error ('..code..') trying to retreive match: '..stringx.shorten(cjson.encode(res), 100))
        return
    end

    if res.matchId and res.region then
        file.write(opt.outdir..path.sep..res.region..path.sep..res.matchId..'.json', cjson.encode(res))
    end
end

local function onMatchlistResponse(res, code)
    -- Status 200 means success and code being nil means we retrieved the data from the cache
    if code ~= nil and code ~= 200 then
        print('Received error ('..code..') trying to retreive matchlist: '..stringx.shorten(cjson.encode(res), 100))
        return
    end

    if res.matches then
        for _, entry in pairs(res.matches) do
            match:getById(entry.matchId, {callback=onMatchResponse})
        end
    end
end

local function onLeagueResponse(res, code)
    -- Status 200 means success and code being nil means we retrieved the data from the cache
    if code ~= nil and code ~= 200 then
        print('Received error ('..code..') trying to retreive league: '..stringx.shorten(cjson.encode(res), 100))
        return
    end

    local leaguePoints = {}
    for _, entry in pairs(res.entries) do
        leaguePoints[entry.playerOrTeamId] = entry.leaguePoints
        table.insert(leaguePoints, {playerId=entry.playerOrTeamId, leaguePoints=entry.leaguePoints})
        matchlist:getBySummonerId(entry.playerOrTeamId, {
            callback=onMatchlistResponse,
            filters={
                seasons={'SEASON2016'},
                rankedQueues={'RANKED_SOLO_5x5','TEAM_BUILDER_DRAFT_RANKED_5x5'}
            },
        })
    end

    file.write(opt.outdir..path.sep..opt.region..'-league.json', cjson.encode(leaguePoints))
end



print('getting '..opt.region..' league')
league:getLeague(opt.tier, 'RANKED_SOLO_5x5', {callback=onLeagueResponse})
