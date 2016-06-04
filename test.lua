local nn = require('nn')
local torch = require('torch')
require('MSEEmbeddingCriterion')
require('TemporalBatchNormalization')

local precision = 1e-5
local tester = torch.Tester()
local testsuite = torch.TestSuite()

local function criterionJacobianTest1DTable(cri, input0, target)
   -- supposes input is a tensor, which is splitted in the first dimension
   local input = input0:split(1,1)
   for i=1,#input do
      input[i] = input[i][1]
   end
   local eps = 1e-6
   local _ = cri:forward(input, target)
   local dfdx = cri:backward(input, target)
   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(input0)
   local input_s = input0:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input0:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input, target)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input, target)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end
   local centraldiff_dfdx_t = centraldiff_dfdx:split(1,1)
   for i=1,#centraldiff_dfdx_t do
      centraldiff_dfdx_t[i] = centraldiff_dfdx_t[i][1]
   end
   for i=1,#centraldiff_dfdx_t do
      -- compare centraldiff_dfdx with :backward()
      local err = (centraldiff_dfdx_t[i] - dfdx[i]):abs():max()
      tester:assertlt(err, precision, ''..i..': error in difference between central difference and :backward')
   end
end

function testsuite.MSEEmbeddingCriterion()
  local dim = 5
  local batch_size = 1
  local crit = nn.MSEEmbeddingCriterion()
  local v = torch.rand(2,batch_size,dim)
  criterionJacobianTest1DTable(crit,v,torch.Tensor{1})
  criterionJacobianTest1DTable(crit,v,torch.Tensor{-1})

  -- batch, sizeAverage true, jacobian
  dim = 5
  batch_size = 2
  crit = nn.MSEEmbeddingCriterion()
  crit.sizeAverage = true
  v = torch.rand(2,batch_size,dim)
  local t = torch.Tensor(batch_size):random(0,1):mul(2):add(-1)
  criterionJacobianTest1DTable(crit,v,t)

  -- batch, sizeAverage false, jacobian
  crit = nn.MSEEmbeddingCriterion()
  crit.sizeAverage = false
  v = torch.rand(2,batch_size,dim)
  t = torch.Tensor(batch_size):random(0,1):mul(2):add(-1)
  criterionJacobianTest1DTable(crit,v,t)
end

local function testBatchNormalization(moduleName, dim, k)
   local planes = torch.random(1,k)
   local size = { torch.random(2, k), planes }
   for _=1,dim do
      table.insert(size, torch.random(1,k))
   end

   local function jacobianTests(module, input, affine)
      local err = nn.Jacobian.testJacobian(module,input)
      tester:assertlt(err,precision, 'error on state ')

      if affine then
         err = nn.Jacobian.testJacobianParameters(module, input,
                                            module.weight, module.gradWeight)
         tester:assertlt(err,precision, 'error on weight ')

         err = nn.Jacobian.testJacobianParameters(module, input,
                                            module.bias, module.gradBias)
         tester:assertlt(err,precision, 'error on weight ')

         err = nn.Jacobian.testJacobianUpdateParameters(module, input, module.weight)
         tester:assertlt(err,precision, 'error on weight [direct update] ')

         err = nn.Jacobian.testJacobianUpdateParameters(module, input, module.bias)
         tester:assertlt(err,precision, 'error on bias [direct update] ')

         for t,terr in pairs(nn.Jacobian.testAllUpdate(module, input, 'weight', 'gradWeight')) do
            tester:assertlt(terr, precision, string.format(
               'error on weight [%s]', t))
         end

         for t,terr in pairs(nn.Jacobian.testAllUpdate(module, input, 'bias', 'gradBias')) do
            tester:assertlt(terr, precision, string.format('error on bias [%s]', t))
         end
      end
      
      -- IO
      local ferr,berr = nn.Jacobian.testIO(module,input)
      tester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
      tester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')
   end
      
   local input = torch.zeros(table.unpack(size)):uniform()
   local module = nn[moduleName](planes)
   module:training()
   jacobianTests(module, input, true)
   module:evaluate()
   jacobianTests(module, input, true)
   
   -- batch norm without affine transform
   module = nn[moduleName](planes, 1e-5, 0.1, false)
   module:training()
   jacobianTests(module, input, false)
   module:evaluate()
   jacobianTests(module, input, false)
end

function testsuite.TemporalBatchNormalization()
   testBatchNormalization('TemporalBatchNormalization', 1, 8)
end

tester:add(testsuite)
tester:run()
