-- Since graph package doesn't return the table graph
-- create a fake global one to get rid of 'accessing
-- undefined variable' warning
graph = {}
require('graph')

local cjson = require('cjson')
local dataset = require('dataset')
local file = require('pl.file')
local lol = require('lol')
local path = require('pl.path')
local tablex = require('pl.tablex')
local torch = require('torch')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate a corpus from League of Legends data')
cmd:text()
cmd:text('Options')
cmd:option('-datadir','dataset','the directory where the data is stored')
cmd:option('-outfile','corpus.txt','the name of the output file which has the corpus')
cmd:text()

-- parse input params
local params = cmd:parse(arg)

local function generateShuffledCorpus(groupings, corpus)
    -- Since the groupings are sets where order doesn't matter
    -- do 'n' random permutations where 'n' is based on the size
    -- of the category. This isn't perfect, but should give a
    -- reasonable distribution that allows the embedding window
    -- to treat the elements of the same group as reasonably similar.
    for _,group in pairs(groupings) do
        if #group > 1 then
            for _=1,math.ceil(math.log(#group, 2)) do
                local indices = torch.randperm(#group)
                for i=1,#group do
                    local groupId = group[indices[i]]
                    corpus:write(groupId..' ')
                end
                corpus:write('\n')
            end
        end
    end
end

local function generateSubCategories(category)
    --------------------------------------------------------------
    -- This function is a bit complex, so here's an example of the sub
    -- categories given the following category:
    --
    -- 'rFlatHPPoolModPerLevel' - Flat, HPPool, HP, Pool, PerLevel
    --------------------------------------------------------------

    -- Remove any leading 'r'
    category = string.match(category, 'r?(.*)')

    -- Remove usage of the word 'Mod'
    local before, after = string.match(category, '(.*)Mod(.*)')
    if before then
        category = string.format('%s%s', before, after or '')
    end

    local categories = {}
    local auxillaryCategories = {Flat=true, Percent=true, PerLevel=false}
    for cat, starting in pairs(auxillaryCategories) do
        local matchStart, matchEnd = string.find(category, cat)
        if matchStart then
            table.insert(categories, string.lower(cat))
            category = starting and string.sub(category, matchEnd+1) or string.sub(category, matchStart, matchEnd)
        end
    end

    -- Treat the remaining string as a category of it's own
    table.insert(categories, string.lower(category))

    -- Treat each sub word as a category
    local matches = string.gmatch(category, 'r?(%u+[%d%l]*)')
    for w in matches do
        table.insert(categories, string.lower(w))
    end

    return categories
end

local function generateTreeCorpus(trees, corpus)
    local lastNode, skipped
    local function printTree(node)
        if lastNode == node and #node.children <= 1 then
            skipped = true
        else
            corpus:write(node:label()..' ')
        end
        lastNode = node
    end

    for _, tree in ipairs(trees) do
        skipped = false
        lastNode = tree
        tree:dfs(printTree)

        if not skipped then
            corpus:write('\n')
        end

        skipped = false
        lastNode = tree
        tree:bfs(printTree)

        if not skipped then
            corpus:write('\n')
        end
    end
end

local function generateMasteryCorpus(datadir, corpus)
    print('generating mastery corpus...')
    local masteryFile = datadir..path.sep..'masteries.json'
    local data = file.read(masteryFile)
    local ok,masteryInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..masteryFile)
    end

    local trees = {}
    for _, tree in pairs(masteryInfo.tree) do
        local nodes = {}
        local childNodes = {}
        for i=#tree,1,-1 do
            local treeItems = tree[i].masteryTreeItems
            for _,treeItem in ipairs(treeItems) do
                if treeItem ~= cjson.null then
                    local node = graph.Node('mastery'..treeItem.masteryId)
                    for _,child in ipairs(childNodes) do
                        node:add(child)
                    end
                    table.insert(trees, node)
                    table.insert(nodes, node)
                end
            end
            childNodes = nodes
            nodes = {}
        end
    end

    generateTreeCorpus(trees, corpus)
end

local function generateRuneCorpus(datadir, runeCategories)
    print('generating rune corpus...')
    local runeFile = datadir..path.sep..'runes.json'
    local data = file.read(runeFile)
    local ok,runeInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..runeFile)
    end

    for _, rune in pairs(runeInfo.data) do
        local runeId = 'rune'..rune.id
        if rune.tags then
            for _, tag in ipairs(rune.tags) do
                for _, category in ipairs(generateSubCategories(tag)) do
                    local tagCategory = runeCategories[category] or {}
                    table.insert(tagCategory, runeId)
                    runeCategories[category] = tagCategory
                end
            end
        end

        if rune.stats then
            for stat in pairs(rune.stats) do
                for _, category in ipairs(generateSubCategories(stat)) do
                    local statCategory = runeCategories[category] or {}
                    table.insert(statCategory, runeId)
                    runeCategories[category] = statCategory
                end
            end
        end

        if rune.rune then
            local tierCategory = 'tier'..rune.rune.tier
            local tier = runeCategories[tierCategory] or {}
            table.insert(tier, runeId)
            runeCategories[tierCategory] = tier

            local typeCategory = 'type'..rune.rune.type
            local runeType = runeCategories[typeCategory] or {}
            table.insert(runeType, runeId)
            runeCategories[typeCategory] = runeType
        end
    end
end

local function generateItemCorpus(datadir, itemCategories)
    print('generating item corpus...')
    local itemFile = datadir..path.sep..'items.json'
    local data = file.read(itemFile)
    local ok,itemInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..itemFile)
    end

    local itemMetaTags = {}
    for _, treeInfo in pairs(itemInfo.tree) do
        for _,tag in ipairs(treeInfo.tags) do
            if tag ~= '_SORTINDEX' then
                itemMetaTags[tag] = treeInfo.header
            end
        end
    end

    local itemTrees = {}
    for _, item in pairs(itemInfo.data) do
        local itemId = 'item'..item.id
        itemTrees[itemId] = graph.Node(itemId)

        if item.tags then
            for _, tag in ipairs(item.tags) do
                local metaTag = itemMetaTags[tag]
                if metaTag then
                    local metaTagCategory = itemCategories[metaTag] or {}
                    table.insert(metaTagCategory, itemId)
                    itemCategories[metaTag] = metaTagCategory
                end

                for _, category in ipairs(generateSubCategories(tag)) do
                    local tagCategory = itemCategories[category] or {}
                    table.insert(tagCategory, itemId)
                    itemCategories[category] = tagCategory
                end
            end
        end

        if item.stats then
            for stat in pairs(item.stats) do
                for _, category in ipairs(generateSubCategories(stat)) do
                    local statCategory = itemCategories[category] or {}
                    table.insert(statCategory, itemId)
                    itemCategories[category] = statCategory
                end
            end
        end

        if item.group then
            for _, category in ipairs(generateSubCategories(item.group)) do
                local group = itemCategories[category] or {}
                table.insert(group, itemId)
                itemCategories[category] = group
            end
        end
    end

    for _, item in pairs(itemInfo.data) do
        local itemId = 'item'..item.id
        local itemNode = itemTrees[itemId]
        if item.from then
            for _,fromId in ipairs(item.from) do
                itemNode:add(itemTrees['item'..fromId])
            end
        end
    end
end

local function generateChampionCorpus(datadir, corpus)
    print('generating champion corpus...')
    local championFile = datadir..path.sep..'champions.json'
    local data = file.read(championFile)
    local ok,championInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..championFile)
    end

    local championCategories = {}
    for _, champion in pairs(championInfo.data) do
        if champion.tags then
            local championId = 'champion'..champion.id
            for _, tag in ipairs(champion.tags) do
                local tagCategory = championCategories[tag] or {}
                table.insert(tagCategory, championId)
                championCategories[tag] = tagCategory
            end
        end
    end

    print('Champion categories: ')
    generateShuffledCorpus(championCategories, corpus)
end

local function generateSpellCorpus(datadir, corpus)
    print('generating spell corpus...')
    local spellFile = datadir..path.sep..'spells.json'
    local data = file.read(spellFile)
    local ok,spellInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..spellFile)
    end

    local spellModes = {}
    for _, spell in pairs(spellInfo.data) do
        if spell.modes then
            local spellId = 'spell'..spell.id
            for _, mode in ipairs(spell.modes) do
                local modeMode = spellModes[mode] or {}
                table.insert(modeMode, spellId)
                spellModes[mode] = modeMode
            end
        end
    end

    generateShuffledCorpus(spellModes, corpus)
end

local function generateVersionCorpus(datadir, corpus)
    print('generating version corpus...')
    local versionFile = datadir..path.sep..'versions.json'
    local data = file.read(versionFile)
    local ok,versionInfo = pcall(function() return cjson.decode(data) end)
    if not ok then
        error('Unable to load: '..versionFile)
    end

    local versions = {}
    for _, version in ipairs(versionInfo) do
        versions[dataset.versionFromString(version)] = true
    end

    generateShuffledCorpus({tablex.keys(versions)}, corpus)
end

local function generateCorpus(datadir, outfile)
    local timer = torch.Timer()
    local corpus = io.open(datadir..path.sep..outfile, 'w')
    local categories = {}

    print('timer: ', timer:time().real)
    generateMasteryCorpus(params.datadir, corpus)

    print('timer: ', timer:time().real)
    generateRuneCorpus(params.datadir, categories)

    print('timer: ', timer:time().real)
    generateItemCorpus(params.datadir, categories)

    print('timer: ', timer:time().real)
    generateChampionCorpus(params.datadir, corpus, categories)

    print('timer: ', timer:time().real)
    generateSpellCorpus(params.datadir, corpus)

    print('timer: ', timer:time().real)
    generateVersionCorpus(params.datadir, corpus)

    generateShuffledCorpus(categories, corpus) --categories
    generateShuffledCorpus({{'DUO', 'NONE', 'SOLO', 'DUO_CARRY', 'DUO_SUPPORT'}}, corpus) --roles
    generateShuffledCorpus({{'MID', 'TOP', 'JUNGLE', 'BOT'}}, corpus) --lanes
    generateShuffledCorpus({tablex.keys(lol.api.Regions)}, corpus) --regions
    generateShuffledCorpus({{'CHALLENGER', 'MASTER', 'DIAMOND', 'PLATINUM', 'GOLD', 'SILVER', 'BRONZE', 'UNRANKED'}}, corpus) --tiers

    print('done in time (seconds): ', timer:time().real)
end

generateCorpus(params.datadir, params.outfile)
