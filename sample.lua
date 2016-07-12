local dataset = require('dataset')
local nngraph = require('nngraph')
local path = require('pl.path')
local torch = require('torch')
require('nn')

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
cmd:option('-version','6.11','what version to predict')
cmd:option('-region','na','what region to predict')
cmd:option('-tier','CHALLENGER','what tier to predict')
cmd:option('-threshold',0,'the minimum confidence threshold for output')
cmd:option('-format','full','what format to use for output: "full" or "compact" (compact does not include confidence)')
cmd:text()

local opt = cmd:parse(arg)
if opt.debug then
    print('setting debug mode')
    nngraph.setDebug(true)
end

local availableFormats = {full=true, compact=true}
local format = opt.format
if not availableFormats[format] then
    print(format..' is not a valid format.')
    os.exit()
end

print('loading model from '..opt.model)
assert(path.exists(opt.model), 'Cannot find the model')
local model = torch.load(opt.model)
local modelType = torch.type(model)
if modelType == 'table' then
    model = model.model
end
assert(model and torch.typename(model), 'Unable to load model')

model:evaluate()

local sampler = dataset.Sampler(opt.datadir)
local input = torch.FloatTensor(1, 6, sampler:embeddingSize())
input[1][1] = sampler:getLaneVector(opt.lane)
input[1][2] = sampler:getRoleVector(opt.lane, opt.role)
input[1][3] = sampler:getChampionVector(opt.champion)
input[1][4] = sampler:getVersionVector(opt.version)
input[1][5] = sampler:getRegionVector(opt.region)
input[1][6] = sampler:getTierVector(opt.tier)

print('champion: '..opt.champion..', role: '..opt.role..', lane: '..opt.lane)
local prediction = sampler:sample(model, input)
for _, row in ipairs(prediction) do
    if row.confidence > opt.threshold then
        local output = 'datatype: '..row.datatype..', name: '..row.data.name
        if format == 'full' then
            output = output..', confidence: '..row.confidence
        end

        print(output)
    end
end
