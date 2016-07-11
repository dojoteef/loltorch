local dir = require('pl.dir')
local path = require('pl.path')
local torch = require('torch')

local snapshot = {}
local _progress = torch.class('Snapshot.Progress', snapshot)
local _snapshot = torch.class('Snapshot.Snapshot', snapshot)

function _progress:__init(opt)
    self.dir = path.join(opt.snapshots, opt.modeltype)
    dir.makepath(self.dir)

    self.fold = 1
    self.kfolds = opt.kfolds
    self.margin = opt.margin
    self.seed = opt.seed
    self.testratio = opt.testratio
    self.frequency = opt.frequency
end

function _progress:filename()
    return path.join(self.dir, 'progress.t7')
end

function _progress:advance()
    self.fold = self.fold + 1
end

function _progress:load()
    local filename = self:filename()
    if path.exists(filename) then
        local progress = torch.load(filename)
        self.fold = progress.fold
        self.seed = progress.seed
        self.kfolds = progress.kfolds
        self.margin = progress.margin
        self.testratio = progress.testratio
        self.frequency = progress.frequency
    end

    torch.manualSeed(self.seed)
end

function _progress:save()
    local data = {}
    data.fold = self.fold
    data.seed = self.seed
    data.kfolds = self.kfolds
    data.margin = self.margin
    data.testratio = self.testratio
    data.frequency = self.frequency

    torch.save(self:filename(), data)
end

function _snapshot:__init(directory, fold)
    self.dir = path.join(directory, 'fold'..fold)
    dir.makepath(self.dir)

    self.epoch = 0
    self.best = {epoch=0,loss=math.huge}
end

function _snapshot:update(best, epoch, regime, optimState)
    self.best = best
    self.epoch = epoch
    self.regime = regime
    self.optimState = optimState
    self.state = torch.getRNGState()
end

function _snapshot:filename()
    return path.join(self.dir, 'latest.t7')
end

function _snapshot:modelfile(epoch)
    return path.join(self.dir, 'epoch'..epoch..'.t7')
end

function _snapshot:optimfile(epoch)
    return path.join(self.dir, 'optim'..epoch..'.t7')
end

function _snapshot:save()
    local data = {}
    data.best = self.best
    data.epoch = self.epoch
    data.regime = self.regime
    data.state = self.state

    torch.save(self:filename(), data)
    torch.save(self:modelfile(self.epoch), self.model)
    torch.save(self:optimfile(self.epoch), self.optimState)
end

function _snapshot:load(best)
    local filename = self:filename()
    if path.exists(filename) then
        local shot = torch.load(filename)
        assert(shot and shot.best and shot.epoch and shot.regime and shot.state, 'Unable to load '..filename)

        self.best = shot.best
        self.epoch = shot.epoch
        self.regime = shot.regime
        self.state = shot.state

        torch.setRNGState(self.state)
    end

    local epoch = best and self.best.epoch or self.epoch
    filename = self:modelfile(epoch)
    if path.exists(filename) then
        local model = torch.load(filename)
        assert(model, 'Unable to load '..filename)

        self.model = model
    end

    filename = self:optimfile(epoch)
    if path.exists(filename) then
        local optimState = torch.load(filename)
        assert(optimState, 'Unable to load '..filename)

        self.optimState = optimState
    end

    return self
end

return snapshot
