open System
open System.IO
open Neuron
open DataSources
open NeuronTraining
open NeuralLayers



let test_and() =
 let data = File.ReadAllLines("test_set6-and.txt") |> Array.map (fun x -> (x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> float x) ))
 let inputs =  new ResizeArray<float array>()

 let outputs = new ResizeArray<float array>()
 for d in data do
  let inp = d.[.. 1]
  let out = d.[2 .. ]
  inputs.Add(inp)
  outputs.Add(out)
 let inps = Array.ofSeq (inputs)
 let outs = Array.ofSeq (outputs)

 let network = new feed_forward_network(2)

 //hidden (input) layer

 Console.WriteLine(outputs.[0].Length)
 //network.AddTanhLayer(2,280,1,true)

 //output layer
 network.AddTanhLayer(outs.[0].Length,2,[|(-1)|],true,0)
 //network.AddTanhLayer(outs.[0].Length,3,(-1),false)

 //network.AddLayer(outs.[0].Length,3,(-1),false)
 //network.AddSoftMaxLayer(outs.[0].Length,2,(-1),false)
 network.FinalizeNet()
 network.set_input(inps.[0])

 //gradient_descent_perceptron_online  180 network (new SimpleProvider(inps)) (new SimpleProvider(outs))
 //gradient_descent_perceptron 15 network (new SimpleProvider(inps)) (new SimpleProvider(outs))
 rprop_perceptron 15 network (new SimpleProvider(inps)) (new SimpleProvider(outs))
 
 let data = network.ComputeGradient(outs.[0])



 Console.WriteLine("--------------")

 let grad = network.AllNumGrad1(outs.[0]) (inps.[0])
 let str = grad |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("Numerical")
 Console.WriteLine(str)

 let data = network.ComputeGradient(outs.[0])
 let str = data |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("Anaytical")
 Console.WriteLine(str)


 Console.WriteLine("--------------")
 network.set_input([|0.0;0.0|])
 let res = network.Compute()
 for r in res do
  Console.WriteLine(r)
 let a1 = res.[0]
  
 Console.WriteLine("0 0")
 network.set_input([|1.0;1.0|])
 let res = network.Compute()
 for r in res do
  Console.WriteLine(r)
 let a2= res.[0]
 
 if a1=1.0 && a2= -1.0 then
  Console.WriteLine("XOR Test passed")
 else
  Console.WriteLine("XOR Test failed")

let test_xor() =
 let data = File.ReadAllLines("test_set6-1.txt") |> Array.map (fun x -> (x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> float x) ))
 let inputs =  new ResizeArray<float array>()

 let outputs = new ResizeArray<float array>()
 for d in data do
  let inp = d.[.. 1]
  let out = d.[2 .. ]
  inputs.Add(inp)
  outputs.Add(out)
 let inps = Array.ofSeq (inputs)
 let outs = Array.ofSeq (outputs)

 let network = new feed_forward_network(2)

 //hidden (input) layer

 Console.WriteLine(outputs.[0].Length)
 //network.AddTanhLayer(2,280,1,true)

 //output layer
 network.AddTanhLayer(3,2,[|1|],true,0)
 network.AddTanhLayer(outs.[0].Length,3,[|(-1)|],false,1)

 //network.AddLayer(outs.[0].Length,3,(-1),false)
 //network.AddSoftMaxLayer(outs.[0].Length,2,(-1),false)
 network.FinalizeNet()
 network.set_input(inps.[0])

 //gradient_descent_perceptron_online  180 network (new SimpleProvider(inps)) (new SimpleProvider(outs))
 rprop_perceptron 15 network (new SimpleProvider(inps)) (new SimpleProvider(outs))
 let data = network.ComputeGradient(outs.[0])



 Console.WriteLine("--------------")

 let grad = network.AllNumGrad1(outs.[0]) (inps.[0])
 let str = grad |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("Numerical")
 Console.WriteLine(str)

 let data = network.ComputeGradient(outs.[0])
 let str = data |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("Anaytical")
 Console.WriteLine(str)


 Console.WriteLine("--------------")
 network.set_input([|0.0;0.0|])
 let res = network.Compute()
 for r in res do
  Console.WriteLine(r)
 let a1 = res.[0]
  
 Console.WriteLine("0 0")
 network.set_input([|1.0;1.0|])
 let res = network.Compute()
 for r in res do
  Console.WriteLine(r)
 let a2= res.[0]
 
 if a1=1.0 && a2= -1.0 then
  Console.WriteLine("XOR Test passed")
 else
  Console.WriteLine("XOR Test failed")
 
  //GOLD ANSWERS :LAST ERROR 4.074E-28, outputs 1;-1,ALL GRADS ZERO

let test_conv_net() =   
 let data = File.ReadAllLines("test_set_conv1a.txt") |> Array.map (fun x -> (x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> float x) ))
 let inputs =  new ResizeArray<float array>()
 Console.WriteLine (data.Length)
 let outputs = new ResizeArray<float array>()
 for d in data do
  Console.WriteLine(d.Length)
  let inp = d.[0.. d.Length - 2]
  let out = d.[d.Length - 1 .. ]
  inputs.Add(inp)
  outputs.Add(out)
 let inps = Array.ofSeq (inputs)
 let outs = Array.ofSeq (outputs)
 Console.WriteLine("here")
 Console.WriteLine(inps.[0].Length)
 Console.WriteLine(outs.[0].Length)
 
 let network = new Elman_network(3) // two layers, 4 hidden units
 
 network.AddCNNLayer(10,1,1,[|1|],true,0) //1 input + 4 hidden buffer
 network.AddMaxPoolingLayer(10,1,[|4|],false,1)
 
 network.AddCNNLayer(10,2,1,[|3|],true,0) //1 input + 4 hidden buffer
 network.AddMaxPoolingLayer(10,1,[|4|],false,1)
 
 
 network.AddTanhLayer(outs.[0].Length,2,[|(-1)|],false,2)
 network.FinalizeNet()
 rprop_uni_bptt 50 network (new SimpleProvider(inps)) (new SimpleProvider(outs)) (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35)) 0.1 None

 Console.WriteLine("---------------------")
 let test_inps = new SimpleProvider([|[|0.0;0.0;0.0|];[|0.0;0.0;1.0;0.0|];[|1.0;0.0;0.0|];[|0.0;1.0;0.0;0.0|];[|0.0;1.0;1.0;0.0|];[|0.0;0.0;1.0;1.0;0.0;0.0;0.0|];[|0.0;0.0;0.0;1.0|];|])
 let over,startw =   network.forward_pass test_inps  (Array.init (test_inps.Length) (fun i -> if i<35 then i else i - 35))  0
 for i = 0 to test_inps.Length - 1 do
  Console.WriteLine("input = " + (test_inps.[i].[0]).ToString() + " output = " + ((network.Forward_buffer i).[0]).ToString())

 Console.WriteLine("--------------")
 network.set_input (test_inps.[5])
 let grad = network.AllNumGrad1([|1.0|]) (test_inps.[5])
 let str = grad |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("Numerical")
 Console.WriteLine(str)
 
 let data = network.ComputeGradient([|1.0|])
 let str = data |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("Anaytical")
 Console.WriteLine(str)
 
  
let test_elman_rnn() = 
 let data = File.ReadAllLines("test_set1.txt") |> Array.map (fun x -> (x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> float x) ))
 let inputs =  new ResizeArray<float array>()

 let outputs = new ResizeArray<float array>()
 for d in data do
  let inp = d.[0.. 0]
  let out = d.[1 .. ]
  inputs.Add(inp)
  outputs.Add(out)
 let inps = Array.ofSeq (inputs)
 let outs = Array.ofSeq (outputs)
 Console.WriteLine(inps.[0].Length)
 Console.WriteLine(outs.[0].Length)
 
 
 
 
 
 let network = new Elman_network(2) // two layers, 4 hidden units
 
 network.AddRecurrentLayer(4,1,[|1|],true,0) //1 input + 4 hidden buffer
 network.AddRecurrentLayer(4,4,[|2|],false,1)
 network.AddTanhLayer(outs.[0].Length,4,[|(-1)|],false,2)
 network.FinalizeNet()
 
 rprop_uni_bptt 40 network (new SimpleProvider(inps)) (new SimpleProvider(outs)) (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35)) 0.1 None
 // gradients
 Console.WriteLine("Numerical")
 let grz = network.AllNumGradB (new SimpleProvider(inps)) (new SimpleProvider(outs)) (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35)) 0
 let str = grz |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("xxxxxx")
 Console.WriteLine(str)
 Console.WriteLine("xxxxxx")

 Console.WriteLine("BPTT")

 let over,startw =   network.forward_pass (new SimpleProvider(inps))  (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35))  0


 let grad = network.bptt_gradient (new SimpleProvider(outs)) (new SimpleProvider(inps))   (over-1) startw
// let grad,_ = network.Batch_gradient_bptt inps outs (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35))
 let str = grad |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine(str)

 
 
  // test eval
 Console.WriteLine("---------------------")
 let test_inps = new SimpleProvider([|[|0.0|];[|0.0|];[|1.0|];[|1.0|];[|1.0|];[|0.0|]|])
 let over,startw =   network.forward_pass test_inps  (Array.init (test_inps.Length) (fun i -> if i<35 then i else i - 35))  0
 for i = 0 to test_inps.Length - 1 do
  Console.WriteLine("input = " + (test_inps.[i].[0]).ToString() + " output = " + ((network.Forward_buffer i).[0]).ToString())

 //GOLD VALUES Error = 0.842713
 //outputs = -0.27 -0.18 0.067 0.71 0.77 0.092
 

let test_BRNN() =
 let data = File.ReadAllLines("test_set4.txt") |> Array.map (fun x -> (x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> float x) ))
 let inputs =  new ResizeArray<float array>()

 let outputs = new ResizeArray<float array>()
 for d in data do
  let inp = d.[0.. 0]
  let out = d.[1 .. ]
  inputs.Add(inp)
  outputs.Add(out)
 let inps = Array.ofSeq (inputs)
 let outs = Array.ofSeq (outputs)

 let network = new BRNN_Elman(3,1,2,false)

 Console.WriteLine(inps.[0].Length)
 //hidden (input) layer
 //network.AddTanhLayer(3,inps.[0].Length + 3,1,true)

 //output layer
 //network.AddSoftMaxLayer(outs.[0].Length,3,(-1),false)
 //network.AddTanhLayer(outs.[0].Length,3,(-1),false)

 //network.AddLayer(outs.[0].Length,2,(-1),false)
 //network.AddSoftMaxLayer(outs.[0].Length,2,(-1),false)

 //network.FinalizeNet()
 //network.set_input(inps.[0])

 //gradient_descent_perceptron_online  80 network inps outs
 Console.WriteLine("hru")
 //rprop_perceptron 10 network inps outs
 //rprop_elman_bptt 260 network inps outs (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35))
 SGD_rmsprop 500 network  (new SimpleProvider(inps))  (new SimpleProvider(outs)) (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35)) 0.01 None

 Console.WriteLine("hru1")
 Console.WriteLine("Numerical")
 let grz = network.AllNumGradB (new SimpleProvider(inps)) (new SimpleProvider(outs)) (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35)) 0
 let str = grz |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("xxxxxx")
 Console.WriteLine(str)
 Console.WriteLine("xxxxxx")

 Console.WriteLine("BPTT")

 let over,startw =   network.forward_pass (new SimpleProvider(inps))  (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35))  0


 let grad = network.BRNN_gradient (new SimpleProvider(inps)) (new SimpleProvider(outs))  (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35)) (over-1) startw
// let grad,_ = network.Batch_gradient_bptt inps outs (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35))
 let str = grad |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine(str)


 Console.WriteLine (over)
 Console.WriteLine (startw)
 // test eval
 let test_inps = new SimpleProvider([|[|0.0|];[|0.0|];[|1.0|];[|1.0|];[|1.0|];[|0.0|]|])
 let over,startw =   network.forward_pass test_inps  (Array.init (test_inps.Length) (fun i -> if i<35 then i else i - 35))  0
 for i = 0 to test_inps.Length - 1 do
  Console.WriteLine("input = " + (test_inps.[i].[0]).ToString() + "output = " + ((network.Forward_buffer.[i].[0]).ToString()) + " " + (network.Forward_buffer.[i].[1]).ToString())

let test_RCNN() =
 let data = File.ReadAllLines("test_set4-1.txt") |> Array.map (fun x -> (x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> float x) ))
 let inputs =  new ResizeArray<float array>()

 let outputs = new ResizeArray<float array>()
 for d in data do
  let inp = d.[0.. 2]
  let out = d.[3 .. ]
  inputs.Add(inp)
  outputs.Add(out)
 let inps = Array.ofSeq (inputs)
 let outs = Array.ofSeq (outputs)

 let network = new RCNN(3,1,2)

 Console.WriteLine(inps.[0].Length)

 Console.WriteLine("hru")
 //rprop_perceptron 10 network inps outs
 //rprop_elman_bptt 260 network inps outs (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35))
 //SGD_rmsprop 500 network  (new SimpleProvider(inps))  (new SimpleProvider(outs)) (Array.init (inps.Length) (fun i -> if i<20 then i else i - 20)) 0.001 None
 
 rprop_uni_bptt 1900 network  (new SimpleProvider(inps))  (new SimpleProvider(outs)) (Array.init (inps.Length) (fun i -> if i<20 then i else i - 20)) 0.01 None

 Console.WriteLine("hru1")
 Console.WriteLine("Numerical")
 let grz = network.AllNumGradB (new SimpleProvider(inps)) (new SimpleProvider(outs)) (Array.init (inps.Length) (fun i -> if i<20 then i else i - 20)) 0
 let str = grz |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine("xxxxxx")
 Console.WriteLine(str)
 Console.WriteLine("xxxxxx")

 Console.WriteLine("BPTT")

 let over,startw =   network.forward_pass (new SimpleProvider(inps))  (Array.init (inps.Length) (fun i -> if i<20 then i else i - 20))  0


 let grad = network.RCNN_gradient (new SimpleProvider(inps)) (new SimpleProvider(outs))  (Array.init (inps.Length) (fun i -> if i<20 then i else i - 20)) (over-1) startw
// let grad,_ = network.Batch_gradient_bptt inps outs (Array.init (inps.Length) (fun i -> if i<35 then i else i - 35))
 let str = grad |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine(str)


 Console.WriteLine (over)
 Console.WriteLine (startw)
 // test eval
 let test_inps = new SimpleProvider([|[|0.0;0.0;0.0|];[|0.0;0.0;0.0|];[|1.0;0.0;0.0|];[|1.0;0.0;0.0|];[|1.0;0.0;0.0|];[|0.0;0.0;0.0|]|])
 //let test_inps = new SimpleProvider([|[|0.0;1.0;1.0|];[|0.0;1.0;1.0|];[|1.0;1.0;1.0|];[|1.0;1.0;1.0|];[|1.0;1.0;1.0|];[|0.0;1.0;1.0|]|])

 let over,startw =   network.forward_pass test_inps  (Array.init (test_inps.Length) (fun i -> if i<20 then i else i - 20))  0
 for i = 0 to test_inps.Length - 1 do
  Console.WriteLine("input = " + (test_inps.[i].[0]).ToString() + "output = " + ((network.Forward_buffer.[i].[0]).ToString()) + " " + (network.Forward_buffer.[i].[1]).ToString())
  
  
  
  
let test_dropout_layer () = 
 let l = new dropout_layer(5,5)
 l.set_input([|2.3;3.2;1.0;2.0;4.8|])
 let r = l.LastResult
 let str = r |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine(str)
 let r = l.full_gradient_neuron ([|2.3;3.2;1.0;2.0;4.8|])
 let berr = l.backprop_error (r)
 let str = berr |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine(str)
 
let test_projection_layer() = 
 let l = new projection_layer(5,0,5)
 l.set_input([|0.0|])
 let output = l.Outputs
 let str = output |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine(str)

 l.set_input([|1.0|])
 let output = l.Outputs
 let str = output |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
 Console.WriteLine(str)
 
 Console.WriteLine("----")
 Console.WriteLine(l.getWeigth 0)
 Console.WriteLine(l.getWeigth 2)
 Console.WriteLine(l.getWeigth 4)
 Console.WriteLine(l.getWeigth 5)
 Console.WriteLine(l.getWeigth 6)
 
  
  
let () =
 //test_xor()
 //test_elman_rnn()
 //test_BRNN()
 //test_dropout_layer()
 // test_projection_layer()
 //test_and()
//test_conv_net()
  test_RCNN()