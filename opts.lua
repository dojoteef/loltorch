local nngraph = require('nngraph')
local torch = require('torch')

local _opts = {}
function _opts.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Train a model which predicts which champion builds are most successful')
    cmd:text()
    cmd:text('Options')
    -----------------------------------------------------------
    -- Training options
    -----------------------------------------------------------
    cmd:option('-batchsize',128,'how many inputs per batch')
    cmd:option('-epochs',200,'max number of to process')
    cmd:option('-patience',5,'how many epochs after finding minimum loss before moving to next training regime')
    cmd:option('-strategy',3,'how aggressive the learning rate, momentum, etc should be (value 1-3)')
    cmd:option('-margin',.25,'what margin to use for the cosine criterion')
    cmd:option('-kfolds',10,'how many folds (must be >= 2)')
    cmd:option('-cross',false,'enable k-fold cross validation (useful for picking model & free parameters)')
    cmd:option('-frequency',0,'update parameters after every x batches processed (0 means use number of threads as update frequency)')
    -----------------------------------------------------------
    -- Data options
    -----------------------------------------------------------
    cmd:option('-testratio',.1,'how many epochs to process')
    cmd:option('-datadir','dataset','the directory where the dataset is located')
    cmd:option('-snapshots','snapshots','where to save snapshots while training to allow for continuing')
    cmd:option('-seed',torch.initialSeed(),'seed used for shuffling data')
    -----------------------------------------------------------
    -- Model options
    -----------------------------------------------------------
    cmd:option('-model','model.t7','where to name the final model file')
    cmd:option('-modeltype','mlp','what type of model to create')
    cmd:option('-models','models','where to store the fully trained models')
    cmd:option('-continue','','where to load a previous model from to continue training')
    -----------------------------------------------------------
    -- General options
    -----------------------------------------------------------
    cmd:option('-debug',false,'whether to start debugguing when an error occurs')
    cmd:option('-verbose',false,'whether to have verbose output')
    cmd:option('-threads',8,'how many threads to use for training')
    cmd:text()

    local opts = cmd:parse(arg or {})
    if opts.debug then
        print('Setting debug mode')
        nngraph.setDebug(true)
    end

    opts.frequency = opts.frequency > 0 and opts.frequency or opts.threads

    return opts
end

return _opts
