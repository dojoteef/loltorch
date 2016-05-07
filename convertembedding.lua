local path = require('pl.path')
local torch = require('torch')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert a word2vec embedding into a Torch friendly format')
cmd:text()
cmd:text('Options')
cmd:option('-datadir','dataset','the directory where the data is stored')
cmd:option('-outfile','embeddings.t7','the name of the output file which has the corpus')
cmd:text()

-- parse input params
local params = cmd:parse(arg)

local function convertEmbeddings(datadir, outfile)
    local embeddingFile = io.open(datadir..path.sep..'embeddings.txt', 'r')
    local numEntries = embeddingFile:read('*n')
    local embeddingSize = embeddingFile:read('*n')
    embeddingFile:read('*l')

    print('Entry Count: '..numEntries..', Embedding Size: '..embeddingSize)

    local embeddings = {}
    for line in embeddingFile:lines() do
        local name
        local i = 1
        local vector = torch.FloatTensor(embeddingSize)
        for chunk in string.gmatch(line, '(%g+)') do
            if not name then
                name = chunk
            else
                vector[i] = tonumber(chunk)
                i = i + 1
            end
        end

        embeddings[name] = vector
    end
    embeddingFile:close()

    torch.save(datadir..path.sep..outfile, embeddings)
end

convertEmbeddings(params.datadir, params.outfile)
