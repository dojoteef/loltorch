local threads = require('threads')
local torch = require('torch')

local utils = {}

function utils.collectAllGarbage()
    local mem
    while mem ~= collectgarbage('count') do
        mem = collectgarbage('count')
        collectgarbage()
    end

    return mem
end

function utils.sliceIndices(index, slices, list)
    local listCount = #list
    local countPerSlice = math.ceil(listCount/slices)
    return ((index-1)*countPerSlice)+1, math.min(listCount, index*countPerSlice)
end

function utils.normalize(tensor, p, dim)
    local norm = torch.norm(tensor, p, dim)
    norm:add(1e-10) -- to prevent divide by zero
    tensor:cdiv(norm:expandAs(tensor))
end

function utils.parseTime(seconds)
    local time = {
        days=0,
        hours=0,
        minutes=0,
        seconds=seconds
    }

    time.days = time.seconds/(24*60*60)
    time.hours = math.fmod(time.days, 1)*24
    time.minutes = math.fmod(time.hours, 1)*60
    time.seconds = math.fmod(time.minutes, 1)*60

    time.days = math.floor(time.days)
    time.hours = math.floor(time.hours)
    time.minutes = math.floor(time.minutes)
    time.seconds = math.floor(time.seconds)

    return time
end

function utils.formatTime(timer)
    local formatted = ''
    local time = utils.parseTime(timer:time().real)
    for _,t in ipairs({'seconds', 'minutes', 'hours', 'days'}) do
        if time[t] > 0 then
            if #formatted > 0 then
                formatted = string.format('%d%s %s', time[t], t:sub(1,1), formatted)
            else
                formatted = string.format('%d%s', time[t], t:sub(1,1))
            end
        end
    end

    return formatted
end

local lastProgress
local stdout = io.output()
function utils.clearProgress()
    if lastProgress then
        stdout:write(string.format('%s%c', string.rep(' ', string.len(lastProgress)), 13))
        stdout:flush()

        lastProgress = nil
    end
end

function utils.progressHeader()
    stdout:write(string.format('%-12s %5s %8s %9s %12s %15s %4s\n', '', 'Epoch', 'Time', 'Progress', 'Batches/Sec', 'Loss', 'Best'))
    stdout:flush()
end

function utils.updateProgress(progress)
    -- Update progress at .01% precision
    local index = progress.index
    local batchCount = progress.batchCount
    if not progress.flush and index ~= batchCount and math.floor(math.fmod(index-1, batchCount/1000)) ~= 0 then
        return
    end

    -- Clear any previous progress
    utils.clearProgress()

    local epoch = progress.epoch and tostring(math.floor(progress.epoch)) or ''
    local action = progress.action
    local percent = index/batchCount * 100
    local persec = index/progress.timer:time().real
    local loss = string.format('%.8f', progress.loss/index)
    local time = utils.formatTime(progress.timer)
    local best = progress.best and 'Y' or ''
    local flush = progress.flush and '\n' or string.char(13)

    lastProgress = string.format('%-12s %5s %8s %8.2f%% %12.2f %15s %4s%s', action, epoch, time, percent, persec, loss, best, flush)
    stdout:write(lastProgress)
    stdout:flush()
end

function utils.createMutexes(num)
    local mutexIds = {}
    for _=1,num do
        local mutex = threads.Mutex()
        table.insert(mutexIds, mutex:id())
    end

    return mutexIds
end

function utils.freeMutexes(mutexIds)
    for _,mutexId in ipairs(mutexIds) do
        local mutex = threads.Mutex(mutexId)
        mutex:free()
    end
end

function utils.createConditions(num)
    local conditionIds = {}
    for _=1,num do
        local condition = threads.Condition()
        table.insert(conditionIds, condition:id())
    end

    return conditionIds
end

function utils.freeConditions(conditionIds)
    for _,conditionId in ipairs(conditionIds) do
        local condition = threads.Condition(conditionId)
        condition:free()
    end
end

return utils
