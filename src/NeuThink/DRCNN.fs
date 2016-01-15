module DRCNN
open System
open System.IO
open System.Collections.Generic
open timeFunction
open DataSources
open NeuralLayers
open Neuron


type DRCNN(context_size:int,input_size:int,output_size:int) =
 let mutable forward_buffer = [|[|0.0|];[|0.0|]|]
 let mutable output_buffer = [|[|0.0|];[|0.0|]|]
 let mutable backprop_buffer = [|[|0.0|];[|0.0|]|]
 let  mutable convnet_buffer = [|0.0|]
 let cnn_output_size = 16

 let recurrent_net = 
  let network = new Elman_network(2)
  network.AddRecurrentLayer(context_size,input_size+cnn_output_size*2,[|(1)|],true,0)
 //network.AddTanhLayer(output_size,context_size,[|(-1)|],false,1)
  network.AddSoftMaxLayer(output_size,context_size,[|(-1)|],false,1)
  network.FinalizeNet()
  network 
  
 let convnet = 
  let network = new Elman_network(3)
 //Console.WriteLine(input_size)
  let total_maps = 50
  let max_layer = 50 * 2
  (*for i = 0 to 24 do
   network.AddCNNLayer(80*input_size,input_size,input_size,[|i*2+1|],true,0,activation_relU) //1 input + 4 hidden buffer
   network.AddMaxPoolingLayer(100,1,[|max_layer|],false,1)
  
  for i = 25 to 49 do
   network.AddCNNLayer(80*input_size,input_size*2,input_size,[|i*2+1|],true,0,activation_relU) //1 input + 4 hidden buffer
   network.AddMaxPoolingLayer(100,1,[|max_layer|],false,1)*)
  
  network.AddCNNLayer(80*input_size,input_size,input_size,[|1|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size,input_size,[|3|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)
  
  network.AddCNNLayer(80*input_size,input_size,input_size,[|5|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size,input_size,[|7|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size*2,input_size,[|9|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)      

  network.AddCNNLayer(80*input_size,input_size*2,input_size,[|11|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size*2,input_size,[|13|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size*2,input_size,[|15|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)     
  
  network.AddCNNLayer(80*input_size,input_size,input_size,[|17|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size,input_size,[|19|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)
  
  network.AddCNNLayer(80*input_size,input_size,input_size,[|21|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size,input_size,[|23|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size*2,input_size,[|25|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)      

  network.AddCNNLayer(80*input_size,input_size*2,input_size,[|27|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size*2,input_size,[|29|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)

  network.AddCNNLayer(80*input_size,input_size*2,input_size,[|31|],true,0) //1 input + 4 hidden buffer
  network.AddMaxPoolingLayer(100,1,[|32|],false,1)    

  //Console.WriteLine(network.Layers.Count)
  //network.AddTanhLayer(cnn_output_size,50,[|101|],false,2) 
 // network.AddTanhLayer(cnn_output_size,50,[|(-1)|],false,2) 
  
  //network.AddDropOutLayer(cnn_output_size,50,[|(-1)|],false,3) 
  
  //network.AddCopyLayer(cnn_output_size,[|(-1)|],false,2)
  network.AddTanhLayer(cnn_output_size,16,[|(-1)|],false,2) 
  network.FinalizeNet()
  network  
  
 member this.Forward_buffer with get() =  forward_buffer and set x = forward_buffer <- x
 member this.ComputeErrorVal (outputs:IOutputProvider) cur_index over_index =
  let mutable errorx = 0.0
  for i = cur_index to over_index do
   let err = Array.sum (Array.map2 squared_error_value_func (this.Forward_buffer.[i-cur_index])  (outputs.[i]))
   errorx <- errorx + err
  errorx
 
 member this.WeightsSize() = 
   recurrent_net.WeightsSize() + convnet.WeightsSize() 
   
 member this.getWeight i = 
   if i < recurrent_net.WeightsSize() then 
    recurrent_net.getWeight i
   else
    convnet.getWeight (i-recurrent_net.WeightsSize())
   

 member this.setWeight i x = 
   if i < recurrent_net.WeightsSize() then 
    recurrent_net.setWeight i x
   else
    convnet.setWeight (i-recurrent_net.WeightsSize()) x
      
 member this.condense_input (inputs:IInputProvider) (indexes:int array) (cur_index:int)=
  let mutable ind = cur_index
  //Console.WriteLine(inputs.[cur_index].Length)
  let z = inputs.[cur_index].[input_size..]
  convnet_buffer <- z
  z
  
  (*let buffer = new ResizeArray<float>()
  
  while (ind < indexes.Length) && (indexes.[ind] <> 0 || cur_index = ind) do
  // Console.WriteLine(ind)
   buffer.AddRange(inputs.[ind])
   ind <- ind + 1
   
  Array.ofSeq(buffer)*)
   
 
 member this.run_recnet (inputs:IInputProvider) (indexes:int array) (start:int) (addin:float array) =
  let mutable ind = start
 
  let mutable new_error = 0.0
  
  while (ind < indexes.Length) && (indexes.[ind] <> 0 || start = ind) do
  // Console.WriteLine(ind)
   let buffer_ind = ind - start
  
   recurrent_net.setStep buffer_ind
   let input = (Array.append (inputs.[ind].[0..input_size - 1] ) (addin)) 
  // Console.WriteLine(input.Length)
  // Console.WriteLine(addin.[0])
  // Console.WriteLine(addin.[4])
   recurrent_net.set_input input
   
   let data = recurrent_net.Compute()
    
    //copy output
   for k=0 to data.Length - 1 do
     this.Forward_buffer.[buffer_ind].[k] <- data.[k]    
   
   
   ind <- ind + 1
   
  ((ind),start)
 
 member this.split_input (inputs:IInputProvider) (cur_index:int) =
  let recword = inputs.[cur_index].[0..input_size-1]
  let ind1 = (int) inputs.[cur_index].[input_size]
 // Console.WriteLine( inputs.[cur_index].[input_size])
  let start_data = inputs.[cur_index].[input_size + 1 .. input_size + ind1]
  let end_data = inputs.[cur_index].[ input_size + ind1 + 1 .. ];
  //Console.WriteLine(end_data.Length)
  (recword,start_data,end_data)
 
 member this.forward_pass (inputs:IInputProvider) (indexes:int array) (start:int) =
   
   this.Forward_buffer <- Array.init 250 (fun x -> Array.init output_size (fun i -> 0.0))
   
   let mutable ind = start
 
   let mutable new_error = 0.0
  
   while (ind < indexes.Length) && (indexes.[ind] <> 0 || start = ind) do
 
    let (recword,start_data,end_data) = this.split_input (inputs:IInputProvider) ind
    let buffer_ind = ind - start 
    //run convnet
    convnet.setStep (buffer_ind * 2)
   // Console.WriteLine(start_data.Length)
    convnet.set_input(start_data)
    let start_part = convnet.Compute()
    
    convnet.setStep ((buffer_ind * 2) + 1)
   // Console.WriteLine(end_data.Length)
    convnet.set_input(end_data)
    let end_part = convnet.Compute()
    
    //end run convnet
    //let resultunt = Array.map2 (fun x y -> x + (-1.0) * y) end_part start_part  
    
    //let resultunt = end_part
    let resultunt = Array.append start_part end_part
    //Console.WriteLine (end_part |> Array.map (fun x -> x.ToString() + " ") |> Array.fold (+) "")
    //Console.WriteLine()
    //Console.WriteLine()
    recurrent_net.setStep buffer_ind
    let input = (Array.append recword resultunt) 
    recurrent_net.set_input input
   
    let data = recurrent_net.Compute()
    
    //copy output
    for k=0 to data.Length - 1 do
      this.Forward_buffer.[buffer_ind].[k] <- data.[k]    
   
   
    ind <- ind + 1
   
   ((ind),start)
   
     
  
   
 member this.full_dataset_eval (inputs:IInputProvider) (outputs:IOutputProvider)  (indexes:int array) =
  let mutable new_error = 0.0
  let mutable i = 0
  while i < inputs.Length - 1 do
   let over,startw =   this.forward_pass inputs  indexes i
   let err = this.ComputeErrorVal outputs startw (over-1)
   new_error <- new_error +  err
   i <- over
  new_error

 member this.BatchError (inputs:IInputProvider) (outputs:IOutputProvider)  (indexes:int array)  =
  this.full_dataset_eval inputs outputs indexes

 member this.Save(filename) =
   let weigths = new ResizeArray<string>()
   for i = 0 to this.WeightsSize() - 1 do
    weigths.Add((this.getWeight i).ToString())
   File.WriteAllLines(filename,(Array.ofSeq weigths))

 member this.Load(filename) =
   let weights  = File.ReadAllLines(filename) |> Array.map (fun x -> float (x.Replace(",",".")))
   for i = 0 to weights.Length - 1 do
    (this.setWeight i (weights.[i]))  
  
 member this.RCNN_gradient (inputs:IInputProvider) (outputs:IOutputProvider)  (indexes:int array) (over:int) (start:int) =
    
    let total_gradient = Array.init (this.WeightsSize())  (fun x -> 0.0)   
    


    for i = (over) downto start do
     let buffer_ind = i - start
              
     recurrent_net.setStep buffer_ind
    
     let grad = recurrent_net.ComputeGradientBptt (outputs.[i]) (recurrent_net.Compute())     
     
     let berrori1 =   recurrent_net.LastBackError.[input_size .. input_size  + cnn_output_size - 1 ]  |> Array.map (fun x ->  -x)
     let berrori2 =   recurrent_net.LastBackError.[input_size + cnn_output_size .. ]  |> Array.map (fun x -> -x)
   
     
     convnet.setStep (buffer_ind*2)
     
     let conv_grad1 = convnet.ComputeGradientBptt berrori1 (([|for j in {0..berrori1.Length-1} -> 0.0|])) 
 
     convnet.setStep ((buffer_ind*2) + 1)
     
     let conv_grad2 = convnet.ComputeGradientBptt berrori2 (([|for j in {0..berrori2.Length-1} -> 0.0|])) 
     
    
  
     
     for j = 0 to grad.Length - 1 do
      total_gradient.[j] <- total_gradient.[j] + grad.[j] 
     for j = 0 to conv_grad1.Length - 1 do
       total_gradient.[j+grad.Length] <- 0.0
       total_gradient.[j+grad.Length] <- total_gradient.[j+grad.Length] + conv_grad1.[j]  
       total_gradient.[j+grad.Length] <- total_gradient.[j+grad.Length] + conv_grad2.[j]
       total_gradient.[j+grad.Length] <- total_gradient.[j+grad.Length] / 2.0      
    
  //   Console.WriteLine(total_gradient.[15+grad.Length])   
   //  Console.WriteLine(conv_grad.[5])
    total_gradient   
    
 member this.Batch_gradient_bptt (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array) =
    
    let new_gradient = Array.init (this.WeightsSize()) (fun x -> 0.0)
    

    let mutable new_error = 0.0
    let mutable i = 0
    while i < inputs.Length - 1 do

     start_profile "fpasee"
     let over,startw =   this.forward_pass inputs  indexes i
     end_profile "fpasee"
     
     let err = this.ComputeErrorVal outputs startw (over-1)
     start_profile "fgragb"
     let grad = this.RCNN_gradient inputs outputs indexes (over-1) startw
     end_profile "fgragb"
     
     new_error <- new_error +  err
     start_profile "gad"
     
     for j = 0 to grad.Length - 1  do
      new_gradient.[j] <- new_gradient.[j] + grad.[j]
     end_profile "gad"
     i <- over

     Console.Write("\r")
     Console.Write("Processing index " + (i.ToString()) + "of total " +  (inputs.Length.ToString()))

    (new_gradient,new_error)
  
 member this.numericalGradB x (input:IInputProvider) (target:IOutputProvider) (indexes:int array) start =
 
   let over,startw = this.forward_pass  input  indexes start
  
   let error = this.ComputeErrorVal target startw (over-1)
   let w = this.getWeight  x
   this.setWeight x (w + 0.000001)
  
   let over1,startw1 = this.forward_pass  input  indexes start
   let error1 =  this.ComputeErrorVal target startw1 (over1-1)
  
   this.setWeight x w
   (error1 - error) / 0.000001

 member this.AllNumGradB (input:IInputProvider) (target:IOutputProvider)  (indexes:int array) (start:int) =
   Array.init (this.WeightsSize()) (fun i -> this.numericalGradB i input target indexes start)
    
 interface ITrainableNetwork with 
   member this.forward_pass inputs indexes i = this.forward_pass  inputs indexes i
   member this.ComputeErrorVal outputs start over = this.ComputeErrorVal outputs start over
   member this.bptt_gradient outputs inputs  indexes start over = this.RCNN_gradient  inputs outputs indexes start over 
   member this.Batch_gradient_bptt inputs outputs indexes = this.Batch_gradient_bptt  inputs outputs indexes
   member this.Save filename = this.Save filename
   member this.setWeight i x = this.setWeight i x
   member this.getWeight x = this.getWeight x 
   member this.WeightsSize() = this.WeightsSize()  
   member this.BatchError x y z = this.BatchError x y z   

