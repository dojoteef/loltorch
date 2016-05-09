require 'nngraph'
local torch = require('torch')

local gModule = torch.getmetatable('nn.gModule')

-- this is redefinition of the share() method for gModule from nnGraph
function gModule:share(gModuleToShare, ...)
  for indexNode, node in ipairs(self.forwardnodes) do
    if node.data.module then
      node.data.module:share(gModuleToShare.forwardnodes[indexNode].data.module, ...)
    end
  end
end
