local dataset = require('dataset')
require('nn')
local nngraph = require('nngraph')
local path = require('pl.path')
local torch = require('torch')
require('MSETableCriterion')
require('TemporalBatchNormalization')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a model which predicts which champion builds are most successful')
cmd:text()
cmd:text('Options')
cmd:option('-model','model.t7','where to save/load the model to/from (if model exists it will be loaded and training will continue on it)')
cmd:option('-datadir','dataset','the directory where the dataset is located')
cmd:option('-debug',false,'whether to set debug mode')
cmd:option('-role','DUO_SUPPORT','what role to predict')
cmd:option('-lane','BOT','what lane to predict')
cmd:option('-champion','Janna','what champion to predict')
cmd:option('-version','6.7','what version to predict')
cmd:option('-region','na','what region to predict')
cmd:option('-tier','MASTER','what tier to predict')
cmd:option('-threshold',.9,'the minimum confidence threshold for output')
cmd:text()

local params = cmd:parse(arg)
if params.debug then
    print('setting debug mode')
    nngraph.setDebug(true)
end

print('loading model from '..params.model)
assert(path.exists(params.model), 'Cannot find the model')
local model = torch.load(params.model)
local modelType = torch.type(model)
if modelType == 'table' then
    model = model.model
end
assert(model and torch.typename(model), 'Unable to load model')

model:evaluate()

local dataLoader = dataset.Loader(params.datadir, 8, 1, 1)
local input = torch.FloatTensor(1, 6, dataLoader:embeddingSize())
input[1][1] = dataLoader:getRoleVector(params.role)
input[1][2] = dataLoader:getChampionVector(params.champion)
input[1][3] = dataLoader:getLaneVector(params.lane)
input[1][4] = dataLoader:getVersionVector(params.version)
input[1][5] = dataLoader:getRegionVector(params.region)
input[1][6] = dataLoader:getTierVector(params.tier)

print('champion: '..params.champion..', role: '..params.role..', lane: '..params.lane)
dataLoader:sample(model, input, params.threshold)
