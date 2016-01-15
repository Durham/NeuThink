namespace NeuThink

open System
open System.IO
open timeFunction
open DataSources
open NeuralLayers
open Neuron

module NeuronTraining =


    let min (x:float) (y:float) =
     if x<y then x else y

    let max (x:float) (y:float) =
     if x>y then x else y

    let sign (x:float) =
     if x>0.0 then 1.0 else -1.0    

(*    let gradient_descent_perceptron num_iters (network:FeedForwardNetwork) (inputs:IInputProvider) (outputs:IOutputProvider) =
     let alpha = 4.0
     for i = 0 to num_iters do
      let gradient,error =  network.Batch_Gradient inputs outputs
      Console.WriteLine("Iteration # " + (i.ToString()) + " Error : " + error.ToString())
      for j = 0 to network.WeightsSize() - 1 do
       network.setWeight j ((network.getWeight j) - (gradient.[j] * alpha))


    let gradient_descent_perceptron_online num_iters (network:FeedForwardNetwork) (inputs:IInputProvider) (outputs:IOutputProvider) =
     let alpha = 0.5
     for i = 0 to num_iters do
      for k = 0 to outputs.Length - 1 do
       let gradient,error =  network.Batch_Gradient((new SimpleProvider([|inputs.[k]|]))) (new SimpleProvider([|outputs.[k]|]))
       for j = 0 to network.WeightsSize() - 1 do
        network.setWeight j ((network.getWeight j) - (gradient.[j] * alpha))
      let gradient,error =  network.Batch_Gradient inputs outputs
      Console.WriteLine("Iteration # " + (i.ToString()) + " Error : " + error.ToString())

       

    let rprop_perceptron  num_iters (network:FeedForwardNetwork) (inputs:IInputProvider) (outputs:IOutputProvider)  =
     Console.WriteLine  network.WeightsSize
     let start_alphas = Array.init  (network.WeightsSize()) (fun x -> 0.1)
     let prev_Gradient=  Array.init  (network.WeightsSize()) (fun x -> 0.0)
     let dMax = 50.0
     let dMin = 0.0000001
     let nplus = 1.2
     let nmin = 0.5
     start_profile "total_rprop"
     for i = 0 to num_iters do
        start_profile "network all"

        let gradient,error =   network.Batch_Gradientinputs outputs

        end_profile "network all"


        start_profile "rprop other"

        Console.WriteLine("Iteration # " + (i.ToString()) + " Error : " + error.ToString())

        for j = 0 to network.WeightsSize() - 1 do

         if  prev_gradient.[j] * gradient.[j] > 0.0 then
          let dw = min (start_alphas.[j] * nplus) dMax
          start_alphas.[j] <- dw
          network.setWeight j ((network.getWeight j) + (- (sign (gradient.[j]))  * dw))
          prev_gradient.[j] <- gradient.[j]

         elif prev_gradient.[j] * gradient.[j] < 0.0 then
          let dw = max(start_alphas.[j] * nmin) dMin
          start_alphas.[j] <- dw
          prev_gradient.[j] <- 0.0

         else
          let dw = start_alphas.[j]

          network.setWeight j ( (network.getWeight j) + (- (sign (gradient.[j])) * dw))

          prev_gradient.[j] <- gradient.[j]

        end_profile "rprop other"

     end_profile "total_rprop"

    let rprop_elman  num_iters (network:RecurrentNetwork) (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array)  =
     Console.WriteLine  network.WeightsSize
     let start_alphas = Array.init  (network.WeightsSize()) (fun x -> 0.005)
     let prev_Gradient=  Array.init  (network.WeightsSize()) (fun x -> 0.0)
     let dMax = 50.0
     let dMin = 0.00000001
     let nplus = 1.2
     let nmin = 0.5
     start_profile "total_rprop"
     for i = 0 to num_iters do
        start_profile "network all"

        let gradient,error =   network.Batch_gradient1 inputs outputs indexes
        end_profile "network all"


        start_profile "rprop other"

        Console.WriteLine("Iteration # " + (i.ToString()) + " Error : " + error.ToString())

        for j = 0 to network.WeightsSize() - 1 do

         if  prev_gradient.[j] * gradient.[j] > 0.0 then
          let dw = min (start_alphas.[j] * nplus) dMax
          start_alphas.[j] <- dw
          network.setWeight j ((network.getWeight j) + (- (sign (gradient.[j]))  * dw))
          prev_gradient.[j] <- gradient.[j]

         elif prev_gradient.[j] * gradient.[j] < 0.0 then
          let dw = max(start_alphas.[j] * nmin) dMin
          start_alphas.[j] <- dw
          prev_gradient.[j] <- 0.0

         else
          let dw = start_alphas.[j]

          network.setWeight j ( (network.getWeight j) + (- (sign (gradient.[j])) * dw))

          prev_gradient.[j] <- gradient.[j]

        end_profile "rprop other"

     end_profile "total_rprop"*)


    let BasicSGD num_iters (network:ITrainableNetwork) (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array)  =

      let mutable new_error = 0.0
      let mutable i = 0
      let alpha = 0.001

      //one SGD PASS
      for j = 0 to num_iters do
       new_error <- 0.0
       while i < inputs.Length - 1 do
         //evalute BPTT
   
         let over,startw =   network.ForwardPass inputs  indexes i
         let err = network.ComputeErrorVal outputs startw (over-1)
         //Console.WriteLine("over=" + over.ToString())
         let grad = network.AllGradient outputs inputs indexes (over-1) startw

         new_error <- new_error +  err
         //UPDATE WEIGHTS
         for j = 0 to network.WeightsSize() - 1 do
          network.setWeight j ((network.getWeight j) - (grad.[j] * alpha))

         i <- over
         //proceed to next stage
         Console.Write("\r")
         Console.Write("Processing index " + (i.ToString()) + "of total " +  (inputs.Length.ToString()))
       Console.WriteLine("Iteration # " + (j.ToString()) + " Error : " + new_error.ToString())
       i <- 0


    let MomentumSGD num_iters (network:ITrainableNetwork) (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array) (start_alpha:float) (callback : (ITrainableNetwork -> unit) option) =

      let mutable new_error = 0.0
      let mutable prev_error = 100000000.0
      let mutable i = 0
      let alpha = 0.7 //viscosity (momentum)
      let mutable lr = start_alpha
      let weights = Array.init (network.WeightsSize()) (fun x ->  network.getWeight x)
      //one SGD PASS

      for j = 0 to num_iters do
       new_error <- 0.0
       while i < inputs.Length - 1 do
         //evalute BPTT

         let over,startw =   network.ForwardPass inputs  indexes i
         let err = network.ComputeErrorVal outputs startw (over-1)
         //Console.WriteLine("over=" + over.ToString())
         let grad = network.AllGradient outputs inputs indexes (over-1) startw

         new_error <- new_error +  err
         //UPDATE WEIGHTS

         for j = 0 to network.WeightsSize() - 1 do
          let dw =   weights.[j] - (network.getWeight j)
          weights.[j] <-  (network.getWeight j)
          network.setWeight j (weights.[j] + (alpha * (dw) - (lr * grad.[j])))

         i <- over
         //proceed to next stage
         Console.Write("\r")
         Console.Write("Processing index " + (i.ToString()) + "of total " +  (inputs.Length.ToString()))
       Console.WriteLine("Iteration # " + (j.ToString()) + " Error : " + new_error.ToString())
       match callback with
          | None -> ignore(None)
          | Some(x) -> x network
       if new_error > prev_error then
        lr <- lr * 0.6
       prev_error <- new_error

       i <- 0


    let grad_norm (grad:float array) = 
     grad |> Array.map (fun x -> x*x) |> Array.sum |> Math.Sqrt
   
   
    let RmspropSGD num_iters (network:ITrainableNetwork) (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array) (start_alpha:float) (callback : (ITrainableNetwork -> unit) option) =
      Console.WriteLine("Time profiled SGD")
      start_profile("total sgd")
      let mutable new_error = 0.0
      let mutable prev_error = 100000.0
      let mutable i = 0
     // let lr = 0.04//25
      //let mutable lr = 0.00015
      //let mutable clip = 0.0001
      let mutable lr = if start_alpha > 0.0 then start_alpha else 0.0015
      let weights = Array.init (network.WeightsSize()) (fun x ->  0.0000000001)
      //one SGD PASS

      for j = 0 to num_iters do
       new_error <- 0.0
       for i  = 0 to weights.Length - 1 do 
        weights.[i] <- 0.0000000001
       while i < inputs.Length - 1 do
         //evalute BPTT

         start_profile("total forward comp")
         let over,startw =   network.ForwardPass inputs  indexes i
         end_profile("total forward comp")

         start_profile("total error comp")
         let err = network.ComputeErrorVal outputs startw (over-1)
         end_profile("total error comp")

         //Console.WriteLine("over=" + over.ToString())
         start_profile("total gradient comp")
         let grad = network.AllGradient outputs inputs   indexes (over-1) startw
         end_profile("total gradient comp")
         //let grad =  network.AllNumGradB  inputs outputs   indexes  startw
     
         //Gradientclipping
     
         (*let norm =grad_norm grad
         if norm > clip then
          let constant = clip / norm
          for i = 0 to grad.Length - 1 do
           grad.[i] <- grad.[i] /  norm   *)  
         start_profile("rms update")
         for i = 0 to weights.Length - 1 do
          weights.[i] <- 0.9 * weights.[i] + 0.1 * (grad.[i] * grad.[i])
         end_profile("rms update")

         new_error <- new_error +  err

         //UPDARE

  
         //UPDATE WEIGHTS
         start_profile("weights update")
         for j = 0 to network.WeightsSize() - 1 do
          //network.setWeight j ((network.getWeight j) - ((grad.[j]/Math.Sqrt(weights.[j])) * lr))
          network.UpdateWeight j ((grad.[j]/Math.Sqrt(weights.[j])) * lr) 
         end_profile("weights update")
         i <- over
         //proceed to next stage
         start_profile("writing text")
         Console.Write("\r")
         Console.Write("Processing index " + (i.ToString()) + "of total " +  (inputs.Length.ToString()))
         end_profile("writing text")
   
       if prev_error<new_error  then
          lr <- lr * 0.9 
          //clip <- clip * 0.02
   
   
       prev_error <- new_error
       Console.WriteLine("Iteration # " + (j.ToString()) + " Error : " + new_error.ToString())
       start_profile("callback")
       match callback with
          | None -> ignore(None)
          | Some(x) -> x network
       end_profile("callback")
       i <- 0
      end_profile("total sgd")
      Console.WriteLine("Time profile:")
      show_time_profile ()

    let RPROP  num_iters (network: ITrainableNetwork) (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array)  (start_alpha:float) (callback : (ITrainableNetwork -> unit) option)  =
     Console.WriteLine  network.WeightsSize
     let start_alphas = Array.init  (network.WeightsSize()) (fun x -> start_alpha)
     let prev_gradient=  Array.init  (network.WeightsSize()) (fun x -> 0.0)
     let dMax = 50.0
     let dMin = 0.000001
     let nplus = 1.2
     let nmin = 0.5 
     start_profile "total_rprop"
     for i = 0 to num_iters do

        let gradient,error =   network.BatchGradient inputs outputs indexes
        //let Gradient= network.AllNumGradB inputs outputs indexes 0
       // let str = Gradient|> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
       // Console.WriteLine(str)

        Console.WriteLine("Iteration # " + (i.ToString()) + " Error : " + error.ToString())
        match callback with
         | None -> ignore(None)
         | Some(x) -> x network
    
        for j = 0 to network.WeightsSize() - 1 do

         if  prev_gradient.[j] * gradient.[j] > 0.0 then
          let dw = min (start_alphas.[j] * nplus) dMax
          start_alphas.[j] <- dw
          let mutable px = ((network.getWeight j) + (- (sign (gradient.[j]))  * dw))
      
          network.setWeight j px
          prev_gradient.[j] <- gradient.[j]

         elif prev_gradient.[j] * gradient.[j] < 0.0 then
          let dw = max(start_alphas.[j] * nmin) dMin
          start_alphas.[j] <- dw
          prev_gradient.[j] <- 0.0

         else
          let dw = start_alphas.[j]
          let mutable px = ( (network.getWeight j) + (- (sign (gradient.[j])) * dw))
      
          network.setWeight j px

          prev_gradient.[j] <- gradient.[j]


