local torch = require('torch')
local MSETableCriterion, parent = torch.class('nn.MSETableCriterion', 'nn.Criterion')

function MSETableCriterion:__init(sizeAverage)
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

function MSETableCriterion:updateOutput(input)
   local input1 = input[1]
   local input2 = input[2]
   self.output_tensor = self.output_tensor or input1.new(1)
   input1.THNN.MSECriterion_updateOutput(
      input1:cdata(),
      input2:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   return self.output
end

function MSETableCriterion:updateGradInput(input)
   local input1 = input[1]
   local input2 = input[2]
   input1.THNN.MSECriterion_updateGradInput(
      input1:cdata(),
      input2:cdata(),
      self.gradInput[1]:cdata(),
      self.sizeAverage
   )
   input2.THNN.MSECriterion_updateGradInput(
      input2:cdata(),
      input1:cdata(),
      self.gradInput[2]:cdata(),
      self.sizeAverage
   )
   return self.gradInput
end
