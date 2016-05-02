local dataset = require('dataset')
require('nn')
local nngraph = require('nngraph')
local path = require('pl.path')
local torch = require('torch')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a model which predicts which champion builds are most successful')
cmd:text()
cmd:text('Options')
cmd:option('-model','model.t7','where to save/load the model to/from (if model exists it will be loaded and training will continue on it)')
cmd:option('-datadir','dataset','the directory where the dataset is located')
cmd:option('-debug',false,'whether to set debug mode')
cmd:option('-role','DUO_SUPPORT','whether to set debug mode')
cmd:option('-lane','BOT','whether to set debug mode')
cmd:option('-champion','Janna','whether to set debug mode')
cmd:text()

local params = cmd:parse(arg)
if params.debug then
    print('setting debug mode')
    nngraph.setDebug(true)
end

print('loading model from '..params.model)
assert(path.exists(params.model), 'Cannot find the model')
local model = torch.load(params.model)

local dataLoader = dataset.Loader(params.datadir, 8, 1, 1)
local role = dataLoader:getRoleIndex(params.role)
local champion = dataLoader:getChampionIndex(params.champion)
local lane = dataLoader:getLaneIndex(params.lane)
local input = torch.Tensor({{role, champion, lane}})

print('champion: '..params.champion..', role: '..params.role..', lane: '..params.lane)
dataLoader:sample(model, input)
