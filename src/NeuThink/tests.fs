let perceptron_test() =
 let data = File.ReadAllLines("test_set2.txt") |> Array.map (fun x -> (x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> float x) ))
 let inputs =  new ResizeArray<float array>()

 let outputs = new ResizeArray<float array>()
 for d in data do
  let inp = d.[0.. d.Length - 3]
  let out = d.[d.Length - 2 .. d.Length - 1]
  inputs.Add(inp)
  outputs.Add(out)
 (*let inps = Array.ofSeq (inputs)
 let outs = Array.ofSeq (outputs)

 let neuron_layer = new perceptron_layer (((outs.[0]).Length),((inps.[0]).Length))

 neuron_layer.Load("layer.txt")
 neuron_layer.set_input([|1.0;0.0|])
 let res = neuron_layer.compute()
 for r in res do
  Console.WriteLine(r)*)

// neuron_layer.train 15 inps outs
// neuron_layer.Save("layer.txt")

let softmax_test() =
 let data = File.ReadAllLines("test_set3.txt") |> Array.map (fun x -> (x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> float x) ))
 let inputs =  new ResizeArray<float array>()

 let outputs = new ResizeArray<float array>()
 for d in data do
  let inp = d.[0.. d.Length - 3]
  let out = d.[d.Length - 2 .. d.Length - 1]
  inputs.Add(inp)
  outputs.Add(out)
 let inps = Array.ofSeq (inputs)
 let outs = Array.ofSeq (outputs)

 let neuron_layer = new softmax_layer (((outs.[0]).Length),((inps.[0]).Length))
 rprop_softmax  10 neuron_layer inps outs

 neuron_layer.set_input([|0.0;0.0|])
 let res = neuron_layer.compute()
 for r in res do
  Console.WriteLine(r)

 (*neuron_layer.Load("layer.txt")
 neuron_layer.set_input([|0.0;0.0|])
 let res = neuron_layer.compute()
 for r in res do
  Console.WriteLine(r)  *)

// neuron_layer.train 15 inps outs
// neuron_layer.Save("layer.txt")


