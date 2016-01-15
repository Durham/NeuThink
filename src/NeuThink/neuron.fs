namespace NeuThink
open System
open System
open System.IO
open System.Collections.Generic
open timeFunction
open DataSources
open NeuralLayers

module Neuron =

//let rnd = new Random(138284)
//let rnd = new Random()



    let square (x:float) = x*x

    let squared_error_gradient_func output actual neuron_derivative  =
     (output - actual) * neuron_derivative

    let squared_error_value_func output actual  =
     square (output - actual)

    //abstract definitions
    ///Interface for all types of neural networks that can be trained with backpropagation
    type ITrainableNetwork =
       // abstract method
       ///Computes forward pass for whole network
       abstract member ForwardPass: IInputProvider -> int array -> int -> (int*int)
       abstract member OutputBuffer: int -> float array
       ///Computes and returns MSE 
       abstract member ComputeErrorVal: IOutputProvider -> int -> int -> float
       
       abstract member AllGradient:  IOutputProvider-> IInputProvider->int array->int->int->float array
       abstract member BatchGradient:  IInputProvider -> IOutputProvider-> int array -> (float array * float)
       abstract member Save: string -> unit
       abstract member Load: string -> unit
       abstract member setWeight: int->float->unit
       abstract member getWeight: int->float
       abstract member UpdateWeight: int->float->unit
       abstract member WeightsSize: unit -> int
       abstract member BatchError: IInputProvider -> IOutputProvider -> (int array) -> float
   
 
 
    // networks 
    /// Base class for all neural networks, 
    /// no support for recurrent layers
    type FeedForwardNetwork()   =
     let layers = new ResizeArray<NeuralLayer>()
     let mutable input_index = 0
     let weights_map = new ResizeArray<int*int>()

     let mutable grad_buffer = [|0.0|]
     let mutable last_back_error = [|0.0|]
     let mutable maxlevel = 0
     let mutable input_buffer =  [|0.0|]
     let mutable comp_buffer =  [|0.0|]

 
     let get_all_level_layers (level:int) = 
      let arr = new ResizeArray<int>()
      for i = 0 to layers.Count - 1 do
       if layers.[i].Level = level then
        arr.Add(i)
      arr
 
     //wide computation
     let  compute_all_poly (input: float array) =
  
       //bottom layers
       let indexes = get_all_level_layers 0
       for index in indexes do
         layers.[index].SetInput(input)
        // Console.WriteLine("set input for" + index.ToString())
   
       //intermidiate layers
   
       for i = 0 to maxlevel - 1 do
        let indexes = get_all_level_layers i
        for index in indexes do
         layers.[index].ComputeEval()
         let connections = layers.[index].Connections
         for cnt in connections do
          layers.[cnt].SetInputUneval(layers.[index].compute())    
          //Console.WriteLine("interim set input for" + cnt.ToString() + "from" + (index.ToString()))      
  
       // output layers  
       //execute layer below
   
       //compute top layer
       let indexes = get_all_level_layers (maxlevel)
       let acc = new ResizeArray<float>()
       for index in indexes do
         layers.[index].ComputeEval()
         //Console.WriteLine("eval is called for " + (index.ToString()))
         //Console.WriteLine((layers.[index].Input.[0].ToString()) + "--" +(layers.[index].Input.[1].ToString()))
         acc.AddRange(layers.[index].compute())
       //Console.WriteLine(acc.[0])
       Array.ofSeq(acc)     
 
 
     let prev_layer i =
      let mutable pl = -1
      for j = 0 to layers.Count - 1 do
       if layers.[j].Connection = i then
        pl <- j
      pl

     let prev_layers i =
      start_profile("layer search")
      let p_layers = new ResizeArray<int>()
      for j = 0 to layers.Count - 1 do
       if layers.[j].Connections.Contains(i) then
        p_layers.Add(j)
      end_profile("layer search")
      p_layers
 
     let add_backprop (index:int) (bdict:Dictionary<int,float array>) (newerr:float array) = 
      if bdict.ContainsKey(index) then
       for i = 0 to newerr.Length - 1 do
        bdict.[index].[i] <- bdict.[index].[i] + newerr.[i]
      else
       bdict.[index] <- Array.init (newerr.Length) (fun x -> newerr.[x]) 
 
 
     let compute_gradient_wide (output:float array)  =
      let mutable windex = 0
      //top layer
      let  berror_l = new Dictionary<int,float array>()
      //Console.WriteLine  (maxlevel)
      let top_layers = get_all_level_layers maxlevel
      for index in top_layers do
       // Console.WriteLine((output.Length.ToString()) + "   " + (index.ToString()) + " " + (layers.[index].Size().ToString()) )
        let grad_neuron =  (layers.[index].GradientNeuron(output))
    
        ignore(layers.[index].Gradient grad_neuron  windex grad_buffer)
        windex <- windex + (layers.[index].WeightsSize)
    
        let berror_ll = (layers.[index].BackpropError  grad_neuron)
        let below_layers = (prev_layers index)
        //below_layers.Reverse()
        let mutable offset = 0
        for lbl in below_layers do
         add_backprop lbl berror_l (berror_ll.[offset..layers.[lbl].Size() + offset - 1])
        // Console.WriteLine((offset.ToString()) + "df" + ((layers.[lbl].Size() + offset - 1).ToString()))
        // Console.WriteLine(lbl.ToString() + "lev" + (berror_l.[lbl].[0].ToString()))
         offset <- offset + (layers.[lbl].Size()) 
    
       // Console.WriteLine (berror_l.[index].Length)
  
      //  
        //distribute berror for layers below
      for layer = maxlevel - 1  downto 0 do
       let indexes = get_all_level_layers  layer
       let mutable offset = 0
       for index in indexes do
        let connected_above = layers.[index].Connections
     //   let p = (berror_l.[connected_above.[0]].Length.ToString()) + " ----- " + (layers.[index].Size().ToString())
     //   Console.WriteLine(p)
      //  let berrorx = (Array.ofSeq connected_above) |> Array.map (fun i -> berror_l.[i]) |> Array.fold (fun a b -> Array.map2 (fun x y -> x+y) a b) (Array.zeroCreate (berror_l.[connected_above.[0]].Length))
      //  Console.WriteLine ( (connected_above.Count.ToString()) + "dffd")
       // Console.WriteLine(offset)
        let grad_neuron =  (layers.[index].GradientNeuron(berror_l.[index]))
      ///  Console.WriteLine("fff" + (berror_l.[index].[0]).ToString())
        ignore(layers.[index].Gradient grad_neuron  windex grad_buffer)
        windex <- windex + layers.[index].WeightsSize
    
    
        let berror_ll = (layers.[index].BackpropError  grad_neuron)
        let below_layers = prev_layers index
        let mutable offset = 0
        for lbl in below_layers do
         add_backprop lbl berror_l (berror_ll.[offset..layers.[lbl].Size() + offset - 1])
        // Console.WriteLine(berror_ll.Length)
         //Console.WriteLine(lbl.ToString() + "levz" + (berror_ll.[0].ToString()))
         offset <- offset + (layers.[lbl].Size()) 
        if below_layers.Count = 0 then
         last_back_error <- Array.init berror_ll.Length (fun x -> berror_ll.[x])
  
      grad_buffer  
    

     ///Compute gradient for network
     member this.ComputeGradientWide (output:float array) = compute_gradient_wide output
 
     member this.FlashInpState () = 
      for i =0 to layers.Count - 1 do
       layers.[i].FlashLastInpIndex() 
   
     ///Provides access to all NN layers
     member this.Layers = layers
     member this.LastBackError = last_back_error
     member this.Prev_layer i = prev_layer i

     member this.Input_index = input_index
   
     ///sets network weight for given index
     member this.setWeight i x =
      //start_profile("NN set weight")
      let layer_index,y = weights_map.[i]
      layers.[layer_index].setWeight y x
      //end_profile("NN set weight")

     member this.UpdateWeight i delta =
      let layer_index,y = weights_map.[i]
      let x = layers.[layer_index].getWeight y 
      layers.[layer_index].setWeight y (x - delta)


     ///return network weight for given index
     member this.getWeight  i =
      let layer_index,y = weights_map.[i]
      layers.[layer_index].getWeight y

     ///returns total number of weights in the network
     member this.WeightsSize() =
      weights_map.Count

     ///Saves all weights to specified filename in text format
     member this.Save(filename) =
      let weigths = new ResizeArray<string>()
      for i = 0 to this.WeightsSize() - 1 do
       weigths.Add((this.getWeight i).ToString())
      File.WriteAllLines(filename,(Array.ofSeq weigths))

     ///Loads all weights from specified filename in text format
     member this.Load(filename) =
      let weights  = File.ReadAllLines(filename) |> Array.map (fun x -> float (x.Replace(",",".")))
      for i = 0 to weights.Length - 1 do
       (this.setWeight i (weights.[i]))
  
     member this.gradBuffer = grad_buffer

     ///Must be called after network structure configuration is finished
     ///before training or computing operations
     member this.FinalizeNet() =
       let mutable index = 0
   
       for layer = maxlevel   downto 0 do
        let indexes = get_all_level_layers  layer
        for index in indexes do
       //   Console.WriteLine((index.ToString()) + "  f " + (layers.[index].WeightsSize.ToString()))
          for j = 0 to layers.[index].WeightsSize - 1 do
           weights_map.Add((index,j))
   
       grad_buffer <- Array.create (this.WeightsSize()) 0.0

     ///sets values for input neurons
     abstract member SetInput : float array -> unit
     default this.SetInput(input: float array) =
      this.FlashInpState ()
      //layers.[input_index].SetInput(input)
      input_buffer <- input
      comp_buffer <- compute_all_poly input
  
     ///forces recomputation of all NN outputs
     member this.forceEval() = 
      comp_buffer <- compute_all_poly input_buffer

     ///<summary>Adds new layer for this neural network</summary>
     ///<param name="layer_size">Number of neurons in the new layer</param>
     ///<param name="input_size">Number of inputs</param>
     ///<param name="connected_with">Array of layers indices,that will accept output from this layer </param>
     ///<param name="is_input">true for bottom layers, false otherwise </param>
     ///<param name="level">specifies the order in which layers should be evaluated</param>
     member this.AddPerceptronLayer(layer_size:int,input_size:int,(connected_with:int array),is_input:bool,level:int,?activation:(float->float)*(float->float) ) =
      let layer = new PerceptronLayer(input_size,layer_size)
      layer.Connections.AddRange(connected_with)
      layer.Level <- level 
      layers.Add (layer)
      if level > maxlevel then maxlevel <- level
      match activation with
       | None -> ignore(None)
       | Some(x) -> layer.setActivation x
      if is_input then
       input_index <- (layers.Count - 1)

     ///Adds new recurrent Elman-type layer for this neural network
     member this.AddRecurrentLayer(layer_size:int,input_size:int,(connected_with:int array),is_input:bool,level:int,?activation:(float->float)*(float->float)) =
      let layer = new RecurrentLayer(input_size+layer_size,layer_size)
    //layer.Connection <- connected_with
      layer.Connections.AddRange(connected_with)
      layer.Level <- level
      layers.Add (layer)
      if level > maxlevel then maxlevel <- level
      match activation with
       | None -> ignore(None)
       | Some(x) -> layer.setActivation x
      if is_input then
       input_index <- (layers.Count - 1) 
       
     member this.AddRecurrentLayerDiag(layer_size:int,input_size:int,(connected_with:int array),is_input:bool,level:int,?activation:(float->float)*(float->float)) =
       let layer = new RecurrentLayer(input_size+layer_size,layer_size)
    //layer.Connection <- connected_with
       layer.Connections.AddRange(connected_with)
       layer.Diagonal()
       layer.Level <- level
       layers.Add (layer)
       if level > maxlevel then maxlevel <- level
       match activation with
        | None -> ignore(None)
        | Some(x) -> layer.setActivation x
       if is_input then
        input_index <- (layers.Count - 1)  
   
     member this.AddDropOutLayer(layer_size:int,input_size:int,(connected_with:int array),is_input:bool,level:int,?droprate) =
      let layer = 
       match droprate with
        | None -> new DropoutLayer(input_size,layer_size,5)
        | Some(x) -> new DropoutLayer(input_size,layer_size,x)
      layer.Connections.AddRange(connected_with)
      layer.Level <- level
      if level > maxlevel then maxlevel <- level
      layers.Add (layer) 
      if is_input then
       input_index <- (layers.Count - 1) 
   
     member this.AddCopyLayer(layer_size:int,(connected_with:int array),is_input:bool,level:int) =
       let layer = new CopyLayer(layer_size)
       layer.Connections.AddRange(connected_with)
       layer.Level <- level
       if level > maxlevel then maxlevel <- level
   
       layers.Add (layer) 
       if is_input then
        input_index <- (layers.Count - 1) 
 
     ///Adds Gaussian noise layer
     member this.AddNoiseLayer(layer_size:int,input_size:int,(connected_with:int array),is_input:bool,nlevel:float,level:int) =
       let layer = new NoiseLayer(input_size,layer_size,nlevel)
       layer.Connections.AddRange(connected_with)
       layer.Level <- level
       if level > maxlevel then maxlevel <- level
   
       layers.Add (layer) 
       if is_input then
        input_index <- (layers.Count - 1) 


     ///Adds one-dimensional convolutional layer
     member this.AddCNNLayer(maxinput:int,filter_size:int,step_size:int,(connected_with:int array),is_input:bool,level:int,?activation:(float->float)*(float->float)) =
       let layer = new ConvlolutionalLayer_1D(maxinput,filter_size,step_size)
       layer.Connections.AddRange(connected_with)
       layer.Level <- level
       for i = 0 to layer.Weights.Length - 1 do
        layer.Weights.[i] <- layer.Weights.[i]  * 8.0
       if level > maxlevel then maxlevel <- level
       match activation with
       | None -> ignore(None)
       | Some(x) -> layer.setActivation x
       layers.Add (layer) 
       if is_input then
        input_index <- (layers.Count - 1) 

     ///Adds one-dimensional max-pooling layer
     member this.AddMaxPoolingLayer(maxinput:int,pool_size:int,(connected_with:int array),is_input:bool,level:int) =
       let layer = new Maxpooling1DLayer(maxinput,pool_size)
       layer.Connections.AddRange(connected_with)
       layer.Level <- level
       if level > maxlevel then maxlevel <- level
       layers.Add (layer) 
       if is_input then
        input_index <- (layers.Count - 1)      
   
     member this.AddProjectionLayer(index_size:int,pass_throw_size:int, proj_size:int,(connected_with:int array),is_input:bool,level:int) =
      let layer = new ProjectionLayer (index_size,pass_throw_size,proj_size)
      layer.Connections.AddRange(connected_with)
      layer.Level <- level
      if level > maxlevel then maxlevel <- level
      layers.Add (layer)
      if is_input then
       input_index <- (layers.Count - 1) 
      layer 
 
     ///Adds softmax output layer
     member this.AddSoftMaxLayer(layer_size:int,input_size:int,(connected_with:int array),is_input:bool,level:int) =
      let layer = new SoftmaxLayer(input_size,layer_size)
      layer.Connections.AddRange(connected_with)
      layer.Level <- level
      layers.Add (layer)
      if level > maxlevel then maxlevel <- level
      if is_input then
       input_index <- (layers.Count - 1)

     ///return last network computation result
     member this.Compute() =
       this.Layers.[this.Layers.Count - 1].LastResult
   
 
     ///computes and returns backpropagation gradient for a given input/output sample
     member this.ComputeGradient(target_output:float array) =
        start_profile "total gradient"
        let output = this.Compute()
        let outgrad = Array.map2(fun y t -> y - t) output target_output
        let start_l = prev_layer (-1)
        let p = compute_gradient_wide  outgrad
        end_profile "total gradient"
        p

     ///computes and returns numerical gradient for the selected weight
     member this.numericalGrad x (target:float array) =
      let def_output = Array.sum (Array.map2 squared_error_value_func (this.Compute())  (target))
      let w = this.getWeight  x
      this.setWeight x (w + 0.000001)
      this.forceEval()
      let new_output = Array.sum (Array.map2 squared_error_value_func (this.Compute())  (target) )
      this.setWeight x w
      (new_output - def_output) / 0.000001

     member this.AllNumGrad (target:float array)  =
      Array.init (this.WeightsSize()) (fun i -> this.numericalGrad i target)

     member this.numericalGrad1 x (target:float array)  (inps:float array) =
      this.SetInput(inps)
      let def_output = Array.sum (Array.map2 squared_error_value_func (this.Compute())  (target))
      let w = this.getWeight  x
      this.setWeight x (w + 0.000001)
      this.SetInput(inps)
      let new_output = Array.sum (Array.map2 squared_error_value_func (this.Compute())  (target) )
      this.setWeight x w
      (new_output - def_output) / 0.000001

     ///computes and returns numerical gradient for the whole network
     member this.AllNumGrad1 (target:float array) (inps:float array) =
      Array.init (this.WeightsSize()) (fun i -> this.numericalGrad1 i target inps) 

     ///computes backpropagation gradient for the whole batch 
     member this.Batch_Gradient(inputs:IInputProvider) (outputs:IOutputProvider)  =
        start_profile("full batch")
        start_profile("grad_createx")
        let new_gradient = Array.init (this.WeightsSize()) (fun x -> 0.0)
        end_profile("grad_createx")

        let mutable new_error = 0.0
        for i=0 to inputs.Length - 1 do
         start_profile("ForwardPass")
         this.SetInput (inputs.[i])
         end_profile("ForwardPass")
         let grad = this.ComputeGradient outputs.[i]

         start_profile("other grad")
         new_error <- new_error +  Array.sum (Array.map2 squared_error_value_func (this.Compute())  (outputs.[i])  )
         for j = 0 to grad.Length - 1  do
          new_gradient.[j] <- new_gradient.[j] + grad.[j]
         end_profile("other grad")
        end_profile("full batch")
        (new_gradient,new_error)


    ///neural network class with support for recurrent layers
    ///can be used both for implementing feed forward and recurrent NNs
    type RecurrentNetwork()   =
     inherit FeedForwardNetwork()
 
     member this.Forward_buffer i = 
      this.setStep i
      this.Compute()
 
     member this.setStep x = 
      for i = 0 to this.Layers.Count - 1 do
       this.Layers.[i].setTimeStep x
 
     member this.ComputeErrorVal (outputs:IOutputProvider) cur_index over_index =
      let mutable errorx = 0.0
      for i = cur_index to over_index do
       this.setStep (i-cur_index)
       let err = Array.sum (Array.map2 squared_error_value_func (this.Compute())  (outputs.[i]))
       if (Double.IsNaN(err)) then
        let str = this.Compute() |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
        Console.WriteLine str
    
        let str = outputs.[i] |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""
        Console.WriteLine str
    
    
       errorx <- errorx + err
      errorx
 
 
     member this.compute_gradient_bptt (output:float array)  =
      this.ComputeGradientWide output
 

     member this.ComputeGradientBptt (target_output:float array) (output: float array) =
        start_profile "total gradient"

        let outgrad = Array.map2(fun y t -> y - t) output target_output
        let p = this.compute_gradient_bptt  outgrad
        end_profile "total gradient"
        p


     member this.numericalGradB x (input:IInputProvider) (target:IOutputProvider) (indexes:int array) start =

      let over,startw = this.ForwardPass  input  indexes start
  
  
      let error = this.ComputeErrorVal target startw (over-1)
      let w = this.getWeight  x
      this.setWeight x (w + 0.000001)
  
      let over1,startw1 = this.ForwardPass  input  indexes start
      let error1 =  this.ComputeErrorVal target startw1 (over1-1)
  
      this.setWeight x w
      (error1 - error) / 0.000001

     member this.AllNumGradB (input:IInputProvider) (target:IOutputProvider)  (indexes:int array) (start:int) =
      Array.init (this.WeightsSize()) (fun i -> this.numericalGradB i input target indexes start)

     override this.SetInput(input: float array) =
      base.SetInput(input)

     member this.Batch_gradient1 (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array) =
        start_profile("full batch")
        let new_gradient = Array.init (this.WeightsSize()) (fun x -> 0.0)
  

        let mutable new_error = 0.0
        for i=0 to inputs.Length - 1 do
         start_profile("ForwardPass")

         this.SetInput (inputs.[i])
         end_profile("ForwardPass")
         let grad = this.ComputeGradient outputs.[i]

         start_profile("other grad")
         new_error <- new_error +  Array.sum (Array.map2 squared_error_value_func (this.Compute())  (outputs.[i])  )
         for j = 0 to grad.Length - 1  do
          new_gradient.[j] <- new_gradient.[j] + grad.[j]
         end_profile("other grad")
        end_profile("full batch")
        (new_gradient,new_error)

     abstract member ForwardPass : IInputProvider -> int array -> int -> int*int
     default this.ForwardPass  (inputs:IInputProvider) (indexes:int array) (cur_index:int) =
      start_profile("forward pass")
      this.FlashInpState ()
      let mutable ind = cur_index
 
      let mutable new_error = 0.0
  
      while (ind < indexes.Length) && (indexes.[ind] <> 0 || cur_index = ind) do
      // Console.WriteLine(ind)
       let buffer_ind = ind - cur_index
  
       this.setStep buffer_ind
       this.SetInput (inputs.[ind])
       let data = this.Compute()
  
       ind <- ind + 1
      end_profile("forward pass")
      ((ind),cur_index)

     abstract member bptt_gradient_zero : IOutputProvider -> IInputProvider -> int -> int -> float array
     default this.bptt_gradient_zero (outputs:IOutputProvider) (inputs:IInputProvider) (cur_index:int) (start_index:int) =
        let pass_grad =  Array.init (this.WeightsSize()) (fun x -> 0.0)
        for i = (cur_index) downto start_index do
         let buffer_ind = i - start_index
    
         this.setStep buffer_ind
         let grad = this.ComputeGradientBptt (outputs.[i]) ([|for i in {0..outputs.[i].Length-1} -> 0.0|])

         for j = 0 to grad.Length - 1  do
          pass_grad.[j] <- pass_grad.[j] + grad.[j]

        pass_grad
 
 
     abstract member bptt_Gradient: IOutputProvider -> IInputProvider -> int -> int -> float array
     default this.bptt_Gradient(outputs:IOutputProvider) (inputs:IInputProvider) (cur_index:int) (start_index:int) =
        start_profile "BATCH GRAD BPTT"
        let pass_grad =  Array.init (this.WeightsSize()) (fun x -> 0.0)
        end_profile "BATCH GRAD BPTT"
    
        for i = (cur_index) downto start_index do
         let buffer_ind = i - start_index
    
         start_profile "compgrad"
         this.setStep buffer_ind
         let grad = this.ComputeGradientBptt (outputs.[i]) (this.Compute())
         end_profile "compgrad"
     
         start_profile "pgrad"
     
         for j = 0 to grad.Length - 1  do
          pass_grad.[j] <- pass_grad.[j] + grad.[j]
         end_profile "pgrad"
     
        pass_grad

     member this.BatchError (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array) =
      let mutable new_error = 0.0
      let mutable i = 0
      while i < inputs.Length - 1 do
         let over,startw =   this.ForwardPass inputs  indexes i
         let err = this.ComputeErrorVal outputs startw (over-1)
         new_error <- new_error +  err
         i <- over
      new_error

     member this.Batch_gradient_bptt (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array) =
    
        let new_gradient= Array.init (this.WeightsSize()) (fun x -> 0.0)
    

        let mutable new_error = 0.0
        let mutable i = 0
        while i < inputs.Length - 1 do

         start_profile "fpasee"
         let over,startw =   this.ForwardPass inputs  indexes i
         end_profile "fpasee"
     
         let err = this.ComputeErrorVal outputs startw (over-1)
         start_profile "fgragb"
         let grad = this.bptt_Gradient outputs inputs (over-1) startw
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
    
     interface ITrainableNetwork with 
      member this.ForwardPass inputs indexes i = this.ForwardPass  inputs indexes i
      member this.ComputeErrorVal outputs start over = this.ComputeErrorVal outputs start over
      member this.AllGradient outputs  inputs indexes start over = this.bptt_Gradient outputs inputs start over 
      member this.BatchGradient inputs outputs indexes = this.Batch_gradient_bptt  inputs outputs indexes
      member this.Save filename = this.Save filename
      member this.setWeight i x = this.setWeight i x
      member this.getWeight x = this.getWeight x 
      member this.UpdateWeight i delta = this.UpdateWeight i delta 
      member this.BatchError x y z = this.BatchError x y z
      member this.WeightsSize() = this.WeightsSize()
      member this.OutputBuffer i = this.Forward_buffer i
      member this.Load filename = this.Load filename
   

    type BackwardRecurrentNetwork ()   =
     inherit RecurrentNetwork()
 
     override this.ForwardPass  (inputs:IInputProvider) (indexes:int array) (cur_index:int) =
      let mutable ind = cur_index
 
      let mutable new_error = 0.0
  
      //find end index
      while (ind < indexes.Length) && (indexes.[ind] <> 0 || cur_index = ind) do
        ind <- ind + 1
    
      let over = ind - 1
      ind <- over 
      // do the pass
      let mutable inc = 0
      while (ind > -1) && (ind <> (cur_index - 1) || over = ind) do
      // Console.WriteLine(ind)
       let buffer_ind = ind - cur_index
  
       this.setStep inc
       this.SetInput (inputs.[ind])
       let data = this.Compute()
   
       ind <- ind - 1
       inc <- inc + 1
   
      ((over + 1),cur_index)

     override this.bptt_Gradient(outputs:IOutputProvider) (inputs:IInputProvider) (cur_index:int) (start_index:int) =
        let pass_grad =  Array.init (this.WeightsSize()) (fun x -> 0.0)
        for i = start_index to cur_index do
        // Console.WriteLine(i)
         let buffer_ind = i - start_index
     
     
         this.setStep buffer_ind
         this.SetInput(inputs.[i])
     
   
         let grad = this.ComputeGradientBptt (outputs.[i]) (this.Compute())
         for j = 0 to grad.Length - 1  do
          pass_grad.[j] <- pass_grad.[j] + grad.[j]
        pass_grad
    
     override this.bptt_gradient_zero (outputs:IOutputProvider) (inputs:IInputProvider) (cur_index:int) (start_index:int) =
        let pass_grad =  Array.init (this.WeightsSize()) (fun x -> 0.0)
        for i = start_index to cur_index do
        // Console.WriteLine(i)
         let buffer_ind = cur_index - (i - start_index)
         //Console.WriteLine(i)
       //  Console.WriteLine(buffer_ind)
         this.setStep buffer_ind
         this.SetInput(inputs.[i])
     
   
         let grad = this.ComputeGradientBptt (outputs.[i]) (([|for i in {0..outputs.[i].Length-1} -> 0.0|]))
         for j = 0 to grad.Length - 1  do
          pass_grad.[j] <- pass_grad.[j] + grad.[j]
        pass_grad 


    type RCNN(context_size:int,input_size:int,output_size:int) =
     let mutable forward_buffer = [|[|0.0|];[|0.0|]|]
     let mutable output_buffer = [|[|0.0|];[|0.0|]|]
     let mutable backprop_buffer = [|[|0.0|];[|0.0|]|]
     let  mutable convnet_buffer = [|0.0|]
     let cnn_output_size = 16

     let recurrent_net = 
      let network = new RecurrentNetwork()
      network.AddRecurrentLayer(context_size,input_size+cnn_output_size,[|(1)|],true,0)
     //network.AddTanhLayer(output_size,context_size,[|(-1)|],false,1)
      network.AddSoftMaxLayer(output_size,context_size,[|(-1)|],false,1)
      network.FinalizeNet()
      network 
  
     let convnet = 
      let network = new RecurrentNetwork()
      //Console.WriteLine(input_size)
  
      network.AddCNNLayer(80*input_size,input_size,input_size,[|1|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size,input_size,[|3|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)
  
      network.AddCNNLayer(80*input_size,input_size,input_size,[|5|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size,input_size,[|7|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size*2,input_size,[|9|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)      

      network.AddCNNLayer(80*input_size,input_size*2,input_size,[|11|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size*2,input_size,[|13|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size*2,input_size,[|15|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)     
  
      network.AddCNNLayer(80*input_size,input_size,input_size,[|17|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size,input_size,[|19|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)
  
      network.AddCNNLayer(80*input_size,input_size,input_size,[|21|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size,input_size,[|23|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size*2,input_size,[|25|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)      

      network.AddCNNLayer(80*input_size,input_size*2,input_size,[|27|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size*2,input_size,[|29|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)

      network.AddCNNLayer(80*input_size,input_size*2,input_size,[|31|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|32|],false,1)     


  
  
  
  
  
      (*network.AddCNNLayer(3*input_size,input_size,input_size,[|1|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|4|],false,1)

      network.AddCNNLayer(3*input_size,input_size,input_size,[|3|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddMaxPoolingLayer(100,1,[|4|],false,1)*)

      (*network.AddCNNLayer(80*input_size,input_size,input_size,[|1|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddCNNLayer(100,2,1,[|2|],true,1,activation_relU)
      network.AddMaxPoolingLayer(100,1,[|24|],false,2)

      network.AddCNNLayer(80*input_size,input_size,input_size,[|4|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddCNNLayer(100,2,1,[|5|],true,1,activation_relU)
      network.AddMaxPoolingLayer(100,1,[|24|],false,2)
  
      network.AddCNNLayer(80*input_size,input_size,input_size,[|7|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddCNNLayer(100,2,1,[|8|],true,1,activation_relU)
      network.AddMaxPoolingLayer(100,1,[|24|],false,2)

      network.AddCNNLayer(80*input_size,input_size,input_size,[|10|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddCNNLayer(100,2,1,[|11|],true,1,activation_relU)
      network.AddMaxPoolingLayer(100,1,[|24|],false,2)
  
      network.AddCNNLayer(80*input_size,input_size,input_size,[|13|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddCNNLayer(100,2,1,[|14|],true,1,activation_relU)
      network.AddMaxPoolingLayer(100,1,[|24|],false,2)

      network.AddCNNLayer(80*input_size,input_size,input_size,[|16|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddCNNLayer(100,2,1,[|17|],true,1,activation_relU)
      network.AddMaxPoolingLayer(100,1,[|24|],false,2)
  
      network.AddCNNLayer(80*input_size,input_size,input_size,[|19|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddCNNLayer(100,2,1,[|20|],true,1,activation_relU)
      network.AddMaxPoolingLayer(100,1,[|24|],false,2)

      network.AddCNNLayer(80*input_size,input_size,input_size,[|22|],true,0,activation_relU) //1 input + 4 hidden buffer
      network.AddCNNLayer(100,2,1,[|23|],true,1,activation_relU)
      network.AddMaxPoolingLayer(100,1,[|24|],false,2)*)
  
    
  
      //network.AddTanhLayer(cnn_output_size,2,[|(-1)|],false,3) 
      //network.AddNoiseLayer(16,16,[|(-1)|],false,0.0,2) 
      //network.AddMaxPoolingLayer(16,16,[|(-1)|],false,2) 
  
      network.AddPerceptronLayer(cnn_output_size,16,[|(-1)|],false,3) 
  
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

     member this.UpdateWeight i delta = 
      if i < recurrent_net.WeightsSize() then 
        recurrent_net.UpdateWeight i delta
       else
        convnet.UpdateWeight (i-recurrent_net.WeightsSize()) delta
    
     
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
       recurrent_net.SetInput input
   
       let data = recurrent_net.Compute()
    
        //copy output
       for k=0 to data.Length - 1 do
         this.Forward_buffer.[buffer_ind].[k] <- data.[k]    
   
   
       ind <- ind + 1
   
      ((ind),start)
 
     member this.ForwardPass (inputs:IInputProvider) (indexes:int array) (start:int) =
   
       this.Forward_buffer <- Array.init 250 (fun x -> Array.init output_size (fun i -> 0.0))
   
       //compute convnet input
       let full_data = this.condense_input inputs indexes start  
       //run convnet and get results
       convnet.SetInput(full_data)
       //Console.WriteLine("  " + (full_data.Length.ToString()) + " ")
       //Console.WriteLine(full_data.[0])
      // Console.WriteLine(full_data.[1]) 
       let cresult = convnet.Compute()
       //Console.WriteLine(cresult.Length)
       //Console.WriteLine("---------")
      // Console.WriteLine(cresult.[0])
       //let cresult = [|0.0;0.0;0.0;0.0;0.0;0.0;0.0;0.0|]
       //convnet_buffer <- cresult
       //create new input for recurrent network
   
       //run recurrent net
   
       let over,start = this.run_recnet inputs indexes start cresult
   
     
       (over,start) 
   
     member this.full_dataset_eval (inputs:IInputProvider) (outputs:IOutputProvider)  (indexes:int array) =
      let mutable new_error = 0.0
      let mutable i = 0
      while i < inputs.Length - 1 do
       let over,startw =   this.ForwardPass inputs  indexes i
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
  
     member this.RCNN_Gradient(inputs:IInputProvider) (outputs:IOutputProvider)  (indexes:int array) (over:int) (start:int) =
    
        let total_gradient= Array.init (this.WeightsSize())  (fun x -> 0.0)   
    


        for i = (over) downto start do
         let buffer_ind = i - start
          //top_network.SetInput(Array.append (forward_net.Forward_buffer (i-start)) (backward_net.Forward_buffer (i-start)))
              
         recurrent_net.setStep buffer_ind
         //recurrent_net.SetInput(Array.append (inputs.[i]) convnet.Compute())
        // Console.WriteLine("rec grad")
         let grad = recurrent_net.ComputeGradientBptt (outputs.[i]) (recurrent_net.Compute())     
        // Console.WriteLine (recurrent_net.LastBackError.Length)  
         //Console.WriteLine (input_size)  
     
         let berrori =   recurrent_net.LastBackError.[input_size .. ]  |> Array.map (fun x -> -x)
     
         convnet.setStep buffer_ind
         convnet.SetInput(convnet_buffer)
        // Console.WriteLine("ZZ")
        // Console.WriteLine(berrori.Length)
        // Console.WriteLine("berr" + (berrori.[2].ToString()))
         let conv_grad = convnet.ComputeGradientBptt berrori (([|for j in {0..berrori.Length-1} -> 0.0|]))     
     
         for j = 0 to grad.Length - 1 do
          total_gradient.[j] <- total_gradient.[j] + grad.[j] 
         for j = 0 to conv_grad.Length - 1 do
          total_gradient.[j+grad.Length] <- total_gradient.[j+grad.Length] + conv_grad.[j]     
    
      //   Console.WriteLine(total_gradient.[15+grad.Length])   
       //  Console.WriteLine(conv_grad.[5])
        total_gradient  
    
     member this.Batch_gradient_bptt (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array) =
    
        let new_gradient= Array.init (this.WeightsSize()) (fun x -> 0.0)
    

        let mutable new_error = 0.0
        let mutable i = 0
        while i < inputs.Length - 1 do

         start_profile "fpasee"
         let over,startw =   this.ForwardPass inputs  indexes i
         end_profile "fpasee"
     
         let err = this.ComputeErrorVal outputs startw (over-1)
         start_profile "fgragb"
         let grad = this.RCNN_Gradient inputs outputs indexes (over-1) startw
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
 
       let over,startw = this.ForwardPass  input  indexes start
  
       let error = this.ComputeErrorVal target startw (over-1)
       let w = this.getWeight  x
       this.setWeight x (w + 0.000001)
  
       let over1,startw1 = this.ForwardPass  input  indexes start
       let error1 =  this.ComputeErrorVal target startw1 (over1-1)
  
       this.setWeight x w
       (error1 - error) / 0.000001

     member this.AllNumGradB (input:IInputProvider) (target:IOutputProvider)  (indexes:int array) (start:int) =
       Array.init (this.WeightsSize()) (fun i -> this.numericalGradB i input target indexes start)
    
     interface ITrainableNetwork with 
       member this.ForwardPass inputs indexes i = this.ForwardPass  inputs indexes i
       member this.ComputeErrorVal outputs start over = this.ComputeErrorVal outputs start over
       member this.AllGradient outputs inputs  indexes start over = this.RCNN_Gradient inputs outputs indexes start over 
       member this.BatchGradient inputs outputs indexes = this.Batch_gradient_bptt  inputs outputs indexes
       member this.Save filename = this.Save filename
       member this.setWeight i x = this.setWeight i x
       member this.getWeight x = this.getWeight x 
       member this.WeightsSize() = this.WeightsSize()  
       member this.BatchError x y z = this.BatchError x y z 
       member this.UpdateWeight i delta = this.UpdateWeight i delta  
       member this.OutputBuffer i = this.Forward_buffer.[i]   
       member this.Load filename = this.Load filename
    
    type BRNN_Elman(context_size:int,input_size:int,output_size:int,dropout:bool) = 
     let mutable forward_buffer = [|[|0.0|];[|0.0|]|]
     let mutable output_buffer = [|[|0.0|];[|0.0|]|]
     let mutable backprop_buffer = [|[|0.0|];[|0.0|]|]

     let forward_net =
      let network = new RecurrentNetwork()
      if dropout then
       //network.AddNoiseLayer(input_size,input_size,[|(1)|],true,0.003,0)
      // network.AddRecurrentLayer(context_size,input_size,1,true )
       //network.AddDropOutLayer(context_size,context_size,2,false)
       //network.AddRecurrentLayer(context_size,context_size,(-1),false)
       network.AddRecurrentLayer(context_size,input_size,[|(-1)|],false,0)
      else
     //  network.AddNoiseLayer(input_size,input_size,(1),true,0.1)
       //network.AddTanhLayer(5,input_size,(1),true)
       //network.AddRecurrentLayer(context_size,input_size,(1),true)
    //    network.AddDropOutLayer(context_size,context_size,2,false)
      
       network.AddPerceptronLayer(30,input_size,[|(3)|],true,0)
       network.AddCopyLayer(50,[|3|],true,0)

       network.AddCopyLayer(50,[|4|],true,0)


       network.AddRecurrentLayerDiag(30,80,[|4|],false,1,activation_relU)

       network.AddPerceptronLayer(context_size,80,[|(-1)|],false,2)


      (* network.AddRecurrentLayer(context_size,input_size,[|(1)|],true,0)
       network.AddRecurrentLayer(context_size,context_size,[|(-1)|],false,1)*)
   
       //network.AddSoftMaxLayer(output_size,context_size,(-1),false)
      network.FinalizeNet()    
      network
   
     let backward_net = 
       let network = new BackwardRecurrentNetwork()
       if dropout then
        //network.AddNoiseLayer(input_size,input_size,[|(1)|],true,0.003,0)
      //  network.AddRecurrentLayer(context_size,input_size,1,true )
      //  network.AddDropOutLayer(context_size,context_size,2,false)
       // network.AddRecurrentLayer(context_size,context_size,(-1),false)
        network.AddRecurrentLayer(context_size,input_size,[|(-1)|],false,0)
       else
       // network.AddRecurrentLayer(context_size,input_size,(1),true )
      //  network.AddNoiseLayer(input_size,input_size,(1),true,0.1)
       // network.AddDropOutLayer(context_size,context_size,2,false)
        (* network.AddRecurrentLayer(context_size,input_size,[|(1)|],true,0)
         network.AddRecurrentLayer(context_size,context_size,[|(-1)|],false,1)*)
         

         
         network.AddPerceptronLayer(30,input_size,[|(3)|],true,0)
         network.AddCopyLayer(50,[|3|],true,0)

         network.AddCopyLayer(50,[|4|],true,0)


         network.AddRecurrentLayerDiag(30,80,[|4|],false,1,activation_relU)

         network.AddPerceptronLayer(context_size,80,[|(-1)|],false,2)

         (*network.AddPerceptronLayer(60,input_size,[|(2)|],true,0)
         network.AddPerceptronLayer(25,input_size,[|(3)|],true,0)

         network.AddRecurrentLayer(30,60,[|3|],false,1,activation_linear)

         network.AddPerceptronLayer(context_size,55,[|(-1)|],false,2)*)

   
        //network.AddRecurrentLayer(context_size,context_size,(-1),false)
        //network.AddRecurrentLayer(context_size,5,(-1),false)
       //network.AddSoftMaxLayer(output_size,context_size,(-1),false)
       network.FinalizeNet()    
       network
  
     let top_network   =
       let network = new RecurrentNetwork()
       if dropout then
       // network.AddDropOutLayer(context_size*2,context_size* 2,1,true)
       // network.AddSoftMaxLayer(output_size,context_size*2,(-1),false) 
        network.AddDropOutLayer(context_size*2,context_size*2,[|1|],false,0)
        network.AddPerceptronLayer(context_size,context_size* 2,[|2|],true,1)
        
        network.AddSoftMaxLayer(output_size,context_size,[|(-1)|],false,2)
   
       else
        network.AddCopyLayer(context_size*2,[|1|],true,0)
        
        network.AddPerceptronLayer(context_size,context_size*2,[|2|],true,1)


        //network.AddDropOutLayer(context_size,context_size,2,false)
        network.AddSoftMaxLayer(output_size,context_size,[|(-1)|],false,2)
   
       //network.AddTanhLayer(context_size,context_size * 2,1,true)
       // network.AddTanhLayer(output_size,context_size * 2,(-1),false)
   
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
       forward_net.WeightsSize() + backward_net.WeightsSize() + top_network.WeightsSize()
   
     member this.getWeight i = 
       if i < forward_net.WeightsSize() then 
        forward_net.getWeight i
       else
        if i > (forward_net.WeightsSize()-1) && (i< forward_net.WeightsSize() + backward_net.WeightsSize()) then
         backward_net.getWeight (i-forward_net.WeightsSize())
        else
          top_network.getWeight (i - (forward_net.WeightsSize() + backward_net.WeightsSize()))

     member this.setWeight i x = 
       if i < forward_net.WeightsSize() then 
        forward_net.setWeight i x
       else
        if i > (forward_net.WeightsSize()-1) && (i< forward_net.WeightsSize() + backward_net.WeightsSize()) then
         backward_net.setWeight (i-forward_net.WeightsSize()) x
        else
          top_network.setWeight (i - (forward_net.WeightsSize() + backward_net.WeightsSize())) x   
   
     member this.UpdateWeight i delta = 
       if i < forward_net.WeightsSize() then 
        forward_net.UpdateWeight i delta
       else
        if i > (forward_net.WeightsSize()-1) && (i< forward_net.WeightsSize() + backward_net.WeightsSize()) then
         backward_net.UpdateWeight (i-forward_net.WeightsSize()) delta
        else
         top_network.UpdateWeight (i - (forward_net.WeightsSize() + backward_net.WeightsSize())) delta  
  
  
     member this.ForwardPass (inputs:IInputProvider) (indexes:int array) (start:int) =
   
       let over,start = forward_net.ForwardPass inputs indexes start
 
       let over1,start1 = backward_net.ForwardPass  inputs indexes start
 
       this.Forward_buffer <- Array.init 250 (fun x -> Array.init output_size (fun i -> 0.0))
   
       for i=start to (over - 1) do
        let buffer_ind = i-start
        top_network.setStep buffer_ind
        top_network.SetInput(Array.append (forward_net.Forward_buffer buffer_ind) (backward_net.Forward_buffer buffer_ind))
        let data = top_network.Compute()
    
        //copy output
        for k=0 to data.Length - 1 do
         this.Forward_buffer.[i-start].[k] <- data.[k]
   
     
     
       (over,start)

     member this.full_dataset_eval (inputs:IInputProvider) (outputs:IOutputProvider)  (indexes:int array) =
      let mutable new_error = 0.0
      let mutable i = 0
      while i < inputs.Length - 1 do
       let over,startw =   this.ForwardPass inputs  indexes i
       let err = this.ComputeErrorVal outputs startw (over-1)
       new_error <- new_error +  err
       i <- over
      new_error

     member this.BatchError (inputs:IInputProvider) (outputs:IOutputProvider)  (indexes:int array)  =
      this.full_dataset_eval inputs outputs indexes
  
     member this.BRNN_Gradient(inputs:IInputProvider) (outputs:IOutputProvider)  (indexes:int array) (over:int) (start:int) =
        start_profile("buuffer")
        backprop_buffer <- Array.init 250 (fun x -> Array.init (context_size*2) (fun i -> 0.0))
        end_profile("buuffer")
        //evalute top layers
        //Console.WriteLine("init")
        let top_grad_sum = Array.init (top_network.WeightsSize()) (fun x -> 0.0)
        for i = start to over do

         //top_network.SetInput(Array.append (forward_net.Forward_buffer (i-start)) (backward_net.Forward_buffer (i-start)))
         top_network.setStep (i-start)
         
         let top_grad = top_network.ComputeGradient outputs.[i]
     
         for t = 0 to top_grad_sum.Length - 1 do 
          top_grad_sum.[t] <- top_grad_sum.[t] + top_grad.[t]
     
         for k =0 to top_network.LastBackError.Length - 1 do 
          backprop_buffer.[i-start].[k] <- top_network.LastBackError.[k]
    
        let berrors = backprop_buffer |> Array.map (fun x -> (x.[0..context_size - 1] |> Array.map (fun x -> -x)))
      
      
        let inputs_x = inputs.[start..over]
      
        let forward_Gradient= forward_net.bptt_gradient_zero (new SimpleProvider(berrors)) inputs_x  (over-start-1) 0
        //Console.WriteLine("Forward_eval")
        //evalute backward layer

    //    for i = 0 to backward_net.Forward_buffer.Length - 1 do
     //    for k =0 to backward_net.Forward_buffer.[i].Length - 1 do
      //    backward_net.Forward_buffer.[i].[k] <- 0.0
   
       //prepare backprop errors
        let berrors = backprop_buffer |> Array.map (fun x -> (x.[0..context_size - 1] |> Array.map (fun x -> -x)))
       
        let inputs_x = inputs.[start..over]
        let backward_Gradient= backward_net.bptt_gradient_zero (new SimpleProvider(berrors)) inputs_x (over-start-1) 0
       //Console.WriteLine("Back evals")
       
        
        let ap = Array.append (Array.append  forward_Gradient backward_Gradient)  top_grad_sum
       
        ap
     interface ITrainableNetwork with 
       member this.ForwardPass inputs indexes i = this.ForwardPass  inputs indexes i
       member this.ComputeErrorVal outputs start over = this.ComputeErrorVal outputs start over
       member this.AllGradient outputs inputs  indexes start over = this.BRNN_Gradient inputs outputs indexes start over 
       member this.BatchGradient inputs outputs indexes = this.Batch_gradient_bptt  inputs outputs indexes
       member this.Save filename = this.Save filename
       member this.setWeight i x = this.setWeight i x
       member this.getWeight x = this.getWeight x 
       member this.UpdateWeight i delta = this.UpdateWeight i delta
       member this.WeightsSize() = this.WeightsSize()   
       member this.BatchError x y z = this.BatchError x y z
       member this.OutputBuffer i = this.Forward_buffer.[i]  
       member this.Load filename = this.Load filename
 
 
     member this.numericalGradB x (input:IInputProvider) (target:IOutputProvider) (indexes:int array) start =
 
       let over,startw = this.ForwardPass  input  indexes start
  
       let error = this.ComputeErrorVal target startw (over-1)
       let w = this.getWeight  x
       this.setWeight x (w + 0.000001)
  
       let over1,startw1 = this.ForwardPass  input  indexes start
       let error1 =  this.ComputeErrorVal target startw1 (over1-1)
  
       this.setWeight x w
       (error1 - error) / 0.000001

     member this.AllNumGradB (input:IInputProvider) (target:IOutputProvider)  (indexes:int array) (start:int) =
       Array.init (this.WeightsSize()) (fun i -> this.numericalGradB i input target indexes start)
  
     member this.Save(filename) =
       let weigths = new ResizeArray<string>()
       for i = 0 to this.WeightsSize() - 1 do
        weigths.Add((this.getWeight i).ToString())
       File.WriteAllLines(filename,(Array.ofSeq weigths))

     member this.Load(filename) =
       let weights  = File.ReadAllLines(filename) |> Array.map (fun x -> float (x.Replace(",",".")))
       for i = 0 to weights.Length - 1 do
        (this.setWeight i (weights.[i]))
   
     member this.Batch_gradient_bptt (inputs:IInputProvider) (outputs:IOutputProvider) (indexes:int array) =
    
        let new_gradient= Array.init (this.WeightsSize()) (fun x -> 0.0)
    

        let mutable new_error = 0.0
        let mutable i = 0
        while i < inputs.Length - 1 do

         start_profile "fpasee"
         let over,startw =   this.ForwardPass inputs  indexes i
         end_profile "fpasee"
     
         let err = this.ComputeErrorVal outputs startw (over-1)
         start_profile "fgragb"
         let grad = this.BRNN_Gradient inputs outputs indexes (over-1) startw
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


  (*  type autoencoder(ninputs:int) =
     inherit FeedForwardNetwork(ninputs)

     member this.Construct(num_hidden:int) =
      //hidden layer
      this.AddTanhLayer(num_hidden,ninputs,[|1|],true,0)
      //output layer
      this.AddTanhLayer(ninputs,num_hidden,[|(-1)|],false,1)
      this.FinalizeNet()

     member this.read_representation() =
      let hidden_layer = this.Layers.[0]
      Array.init hidden_layer.WeightsSize (fun i -> hidden_layer.getWeight i)

     member this.set_representation (data:float array) =
      let output_layer = this.Layers.[1]
      output_layer.SetInput data

     member this.compute_originals() =
      let output_layer = this.Layers.[1]
      output_layer.compute()*)


    
    




