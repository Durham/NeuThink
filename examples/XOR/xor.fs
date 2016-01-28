
open NeuThink.Neuron
open NeuThink.NeuronTraining

let() =
 let nn = new GeneralNetwork()
 nn.AddPerceptronLayer(4,2,[|1|],true,0)
 nn.AddPerceptronLayer(1,4,[|-1|],false,1)
 nn.FinalizeNet()


 let outputs = [|[|-1.0|];[|-1.0|];[|1.0|];[|1.0|]|]
 let inputs = [|[|1.0;1.0|];[|0.0;0.0|];[|1.0;0.0|];[|0.0;1.0|]|]

 MomentumSGD 100 nn (new NeuThink.DataSources.SimpleProvider(inputs)) (new NeuThink.DataSources.SimpleProvider(outputs)) 0.2  (Some([|0;0;0;0|])) None
 nn.SetInput([|1.0;1.0|])
 System.Console.WriteLine(nn.Compute().[0])




