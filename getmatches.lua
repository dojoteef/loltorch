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

local ignoreList
local ignoreFile = path.join(opt.outdir, opt.region, '.ignorelist')
if path.exists(ignoreFile) then
    local data = file.read(ignoreFile)
    local ok,list = pcall(function() return cjson.decode(data) end)
    if ok then
        ignoreList = list
    end
end
ignoreList = ignoreList or {}

local function onMatchResponse(res, code, matchId)
    -- Status 200 means success and code being nil means we retrieved the data from the cache
    if code ~= nil and code ~= 200 then
        if code == 404 then
            -- while this isn't very efficient if there are a large number of missing matches that doesn't
            -- seem to be the case in practice so opting for a quick solution
            ignoreList[tostring(matchId)] = true
            file.write(ignoreFile, cjson.encode(ignoreList))
        end

        print('Received error ('..code..') trying to retreive match: '..stringx.shorten(cjson.encode(res), 100))
        return
    end

    if res.matchId and res.region then
        file.write(path.join(opt.outdir, res.region, res.matchId..'.json'), cjson.encode(res))
    end
end

local function onMatchResponseDecorator(matchId)
    return function (res, code)
        onMatchResponse(res, code, matchId)
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
            if not ignoreList[tostring(entry.matchId)] then
                match:getById(entry.matchId, {callback=onMatchResponseDecorator(entry.matchId)})
            end
        end
    end
end

local function onLeagueResponse(res, code)
    -- Status 200 means success and code being nil means we retrieved the data from the cache
    if code ~= nil and code ~= 200 then
        print('Received error ('..code..') trying to retreive league: '..stringx.shorten(cjson.encode(res), 100))
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
