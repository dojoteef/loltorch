local torch = require('torch')
local MSEEmbeddingCriterion, parent = torch.class('nn.MSEEmbeddingCriterion', 'nn.Criterion')

function MSEEmbeddingCriterion:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
    self.sizeAverage = true
    self.eps = 1e-6
end

function MSEEmbeddingCriterion:updateOutput(input, y)
    local input1 = input[1]
    local input2 = input[2]

    if not self.idx then
        self.ln = input1.new()
        self.mse = input1.new()
        self.buffer = input1.new()
        self.idx = torch.ByteTensor()
    end

    self.buffer:resizeAs(input1):copy(input1):csub(input2)
    self.buffer:pow(2)

    self.mse:sum(self.buffer, 2)
    self.ln:resizeAs(self.mse):copy(self.mse):log1p()

    y.eq(self.idx,y,-1)
    self.ln:resizeAs(self.mse):copy(self.mse):log1p()
    self.ln[self.idx] = self.ln[self.idx]:cmax(self.eps):cinv()

    self.output = self.ln:sum()
    if self.sizeAverage then
        self.output = self.output/y:size(1)
    end

    return self.output
end

function MSEEmbeddingCriterion:updateGradInput(input, y)
    local input1 = input[1]
    local input2 = input[2]

    local gradInput1 = self.gradInput[1]
    local gradInput2 = self.gradInput[2]

    self.mse:add(1):cmax(self.eps):cinv()
    self.mse = self.mse:view(-1,1):expand(input1:size())
    gradInput1:resizeAs(input1):copy(input1):csub(input2):mul(2):cmul(self.mse)
    gradInput2:resizeAs(input2):copy(input2):csub(input1):mul(2):cmul(self.mse)

    y.eq(self.idx,y,-1)
    self.ln[self.idx] = self.ln[self.idx]:pow(2):neg()
    self.ln = self.ln:view(-1,1):expand(input1:size())

    self.idx = self.idx:view(-1,1):expand(gradInput1:size())
    gradInput1[self.idx] = gradInput1[self.idx]:cmul(self.ln[self.idx])
    gradInput2[self.idx] = gradInput2[self.idx]:cmul(self.ln[self.idx])

    if self.sizeAverage then
        gradInput1:div(y:size(1))
        gradInput2:div(y:size(1))
    end

    return self.gradInput
end
