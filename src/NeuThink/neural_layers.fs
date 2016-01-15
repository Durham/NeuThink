namespace NeuThink
open System
open System.IO
open timeFunction



module NeuralLayers =
    let private rnd = new Random(447656776)
    //let rnd = new Random()

    let private rnd_dropout = new Random(65535)
    let private rnd_noise = new Random(65535)


    //math functions
    let private two2one x y size = y*size+x


    

    let private dgemv_cpu  (weights:float array) (proc_inputs:float array) (outputs:float array) =
      let mutable x = 0
      
      let t =  proc_inputs.Length
      let t1 =  proc_inputs.Length - 1

      for i =0 to outputs.Length - 1 do
      // let x = i * (proc_inputs.Length)
       let mutable kx = 0.0
       for j = x to x + t1 do
        kx <- kx + weights.[j] * proc_inputs.[j-x]
       x <- x +  t
       outputs.[i] <- kx

  
    let private convolve (signal:float array) (filter:float array) (signal_x:int,signal_y:int) (filter_x:int,filter_y:int) (stride:int)=
      let outputs_size_x =(signal_x / stride) - ((filter_x / stride) - 1)
      let outputs_size_y =(signal_y / stride) - ((filter_y / stride) - 1)
      let outputs = Array.init (outputs_size_x*outputs_size_y ) (fun x -> 0.0)
      let mutable kx = 0.0
      for i = 0 to outputs_size_y - 1 do
       for j = 0 to outputs_size_x - 1 do
        kx <- 0.0
        for i_f = 0 to filter_y - 1 do
         for j_f = 0 to filter_x - 1 do
          let fs = two2one (j+j_f) (i+i_f) signal_x
          kx <- kx + filter.[two2one j_f i_f filter_x] * signal.[two2one (j+j_f) (i+i_f) signal_x]
        let p = (two2one j i outputs_size_x)
        outputs.[(two2one j i outputs_size_x)] <- kx
      outputs 
   
    let private inv_convolve (signal:float array) (filter:float array) (signal_x:int,signal_y:int) (filter_x:int,filter_y:int) (stride:int)=
      let outputs_size_x =(signal_x / stride) - ((filter_x / stride) - 1)
      let outputs_size_y =(signal_y / stride) - ((filter_y / stride) - 1)
      let outputs = Array.init (outputs_size_x*outputs_size_y ) (fun x -> 0.0)
      let mutable kx = 0.0
      for i = 0 to outputs_size_y - 1 do
       for j = 0 to outputs_size_x - 1 do
        kx <- 0.0
        for i_f = 0 to filter_y - 1 do
         for j_f = 0 to filter_x - 1 do
          let fs = two2one (j+j_f) (i+i_f) signal_x
          kx <- kx + filter.[two2one (filter_x - j_f) (filter_y - i_f) filter_x] * signal.[two2one (j+j_f) (i+i_f) signal_x]
        let p = (two2one j i outputs_size_x)
        outputs.[(two2one j i outputs_size_x)] <- kx
      outputs 
   
    let private conv_grad (input:float array) (error:float array) (input_x,input_y) (filter_x,filter_y) (stride:int) =
        let outputs_size_x =(input_x / stride) - ((filter_x / stride) - 1)
        let outputs_size_y =(input_y / stride) - ((filter_y / stride) - 1)
        let result_grad = Array.init (filter_x*filter_y) (fun x -> 0.0)
        for i = 0 to outputs_size_y - 1 do
         for j = 0 to outputs_size_x - 1 do 
           let cur_error =   error.[(two2one j i outputs_size_x)] 
           for i_f = 0 to filter_y - 1 do
            for j_f = 0 to filter_x - 1 do
              result_grad.[(two2one j_f i_f filter_x)] <- result_grad.[(two2one j_f i_f filter_x)]  +  input.[two2one (j+j_f) (i+i_f) input_x] * cur_error
        result_grad      
              
   
    //end math


    //let rnd = new Random()

    ///Base class for implementing neural network layers
    [<AbstractClass>]
    type NeuralLayer(ninputs:int,noutputs:int) =

     let mutable connected_with = 0
     let last_result =  Array.init noutputs (fun i -> 0.0)
     let connections = new ResizeArray<int>()
     let mutable last_input_index = 0
     let mutable level = 0
     let mutable frozen = false
 
     member this.Connections = connections
     member this.LastInputIndex  with get() = last_input_index and set(x) = last_input_index <- x
     member this.Frozen  with get() =frozen and set(x) = frozen <- x
     member this.FlashLastInpIndex() =  last_input_index <- 0
     member this.Level  with get() = level and set(x) = level <- x
 
     member this.Connection with get() =  connected_with and set(x) =  connected_with <- x

     ///Sets input vector for this layer and computes outputs
     abstract member SetInput : (float array) -> unit
     ///Sets input vector for this layer without computing result
     abstract member SetInputUneval : (float array) -> unit
     ///obtain last output of this layer
     abstract member compute :  unit -> (float array)
     ///returns a number of output neurons
     abstract member Size : unit -> int
     abstract member print_state : unit -> unit
     default this.print_state () = ignore(None)
     abstract member setResult : int -> float -> unit
     ///returns the number of weights 
     abstract member WeightsSize: unit->int with get
     default this.WeightsSize =
         (ninputs + 1) * noutputs

     ///Set the value of specified weight
     abstract member setWeight : int -> float -> unit
     ///returns the value of specified weight
     abstract member getWeight :  int -> float
     ///returns the size of layer gradient
     abstract member gradSize : unit -> int
     ///sets current timestep for layers when using BPTT
     abstract member setTimeStep: int -> unit
     ///force computation of the output
     abstract member ComputeEval: unit -> unit

     abstract member LastResult :  float array
     default this.LastResult = last_result

     ///computes gradient per output neuron
     abstract member GradientNeuron  :(float array) -> (float array)
     ///computes gradient per weight
     abstract member Gradient: (float array) -> int -> (float array) -> (float array)
     ///computes backpropagation error to pass for layer below
     abstract member BackpropError :   (float array) -> (float array)

    ///Implements dropout regularization
    type DropoutLayer(ninputs:int,noutputs:int,droprate:int) =
     inherit  NeuralLayer(ninputs,noutputs)
     let inputs =     Array.init 440 (fun i -> Array.init ninputs (fun x -> 0.0))
     let outputs =  Array.init 440 (fun i -> Array.init noutputs (fun x -> 0.0))
     let drop_pattern = Array.init 440 (fun i -> Array.init noutputs (fun x -> 0.0)) 
     let berror_buffer = Array.init ninputs (fun x -> 0.0)
     let mutable cur_time = 0

     let multiplier =  (10.0 / (float) (10 - droprate))
     ///Provides access to current outputs array using current timestep
     member this.Outputs = outputs.[cur_time]
     ///Provides access to current inputs array using current timestep
     member this.Inputs = inputs.[cur_time]
     member this.DropPattern = drop_pattern.[cur_time]
 
     //let drops = Array.init 100 (fun x -> Array.init
 
     override this.setTimeStep x = cur_time <- x
     override this.WeightsSize = 0
     override this.setWeight x y = ignore(None)
     override this.getWeight i  = 0.0
     override this.gradSize() = 0
     override this.Size() = noutputs
     //override this.GradientNeuron (target:float array) = 
  
     override this.BackpropError (grad:float array) = 
      Array.init berror_buffer.Length (fun i -> berror_buffer.[i])
 
     override this.ComputeEval() = 
      let proc_inputs = this.Inputs
  
      for i  = 0 to  proc_inputs.Length - 1 do
    
        if (rnd_dropout.Next(10)) > (droprate) then
         this.Outputs.[i] <- proc_inputs.[i] * multiplier
         this.DropPattern.[i] <- 1.0 
        else 
         this.Outputs.[i] <- 0.0 
         this.DropPattern.[i] <- 0.0      
  
     override this.Gradient a b c = 
      [||]
  
     override this.LastResult = this.Outputs
 
     override this.GradientNeuron target = 
       let p = target |> Array.mapi (fun i x -> x * this.DropPattern.[i])
       for i = 0 to p.Length - 1 do
        berror_buffer.[i] <-p.[i] / multiplier
       [||] 
 
     override this.setResult  i x = 
      this.Outputs.[i] <- x
 
     override this.SetInput (input:float array) =
      for i = this.LastInputIndex to input.Length - 1 do
       this.Inputs.[i] <- input.[i]
      this.LastInputIndex <-this.LastInputIndex + input.Length
      this.ComputeEval()
  
     override this.SetInputUneval (input:float array) =
      for i =  this.LastInputIndex to input.Length - 1 do
       this.Inputs.[i] <- input.[i]
      this.LastInputIndex <- this.LastInputIndex + input.Length
 
     override this.compute() = 
      this.Outputs

    ///implements one dimensional k-max pooling for simple text processing convnets
    type Maxpooling1DLayer(maxinputs:int,npooling:int) =
     inherit  NeuralLayer(maxinputs,npooling)
     let inputs =     Array.init 440 (fun i -> Array.init maxinputs (fun x -> 0.0))
     let outputs =  Array.init 440 (fun i -> Array.init npooling (fun x -> 0.0))
     let indexes =  Array.init 440 (fun i -> Array.init npooling (fun x -> 0))
     let berror_buffer = Array.init npooling (fun x -> 0.0)
     let mutable cur_time = 0
     ///Provides access to current outputs array using current timestep
     member this.Outputs = outputs.[cur_time]
     ///Provides access to current inputs array using current timestep
     member this.Inputs = inputs.[cur_time]
     member this.Indexes = indexes.[cur_time]
 
     //let drops = Array.init 100 (fun x -> Array.init
 
     override this.setTimeStep x = cur_time <- x
     override this.WeightsSize = 0
     override this.setWeight x y = ignore(None)
     override this.getWeight i  = 0.0
     override this.gradSize() = 0
     override this.Size() = npooling
 
 
     override this.ComputeEval() =  
       let nout = Array.mapi (fun i x -> (x,i)) (this.Inputs)
       let nsorted =nout |> Array.sortBy (fun (x,i) -> 0.0 - (Math.Abs x)) 
       let kmax = nsorted.[0..(npooling-1)]
       for i = 0 to this.Outputs.Length - 1 do
        this.Outputs.[i] <- (fst kmax.[i])
        this.Indexes.[i] <- (snd kmax.[i])
     //  Console.WriteLine ("fds" + ((this.Outputs.[i]).ToString()))
 
     override this.Gradient a b c = 
      [||]
  
     override this.LastResult = this.Outputs
 
     override this.setResult  i x = 
      this.Outputs.[i] <- x
 
     override this.SetInput (input:float array) =
     // Console.WriteLine(input.Length)
    //  Console.WriteLine(this.Inputs.Length)
      for i = this.LastInputIndex  to input.Length - 1 do
       this.Inputs.[i] <- input.[i]
      this.LastInputIndex <- this.LastInputIndex + input.Length
  
      this.ComputeEval()
  
     override this.SetInputUneval (input:float array) =
      for i = this.LastInputIndex to input.Length - 1 do
       this.Inputs.[i] <- input.[i]
      this.LastInputIndex <- this.LastInputIndex + input.Length
 
     override this.compute() = 
      this.Outputs.[0..npooling-1]
  
     override this.GradientNeuron (target:float array) = 
      for i = 0 to target.Length - 1 do
       berror_buffer.[i] <- target.[i]
       //Console.WriteLine("buffer" + (berror_buffer.[i] .ToString()))
      [||]
 
     member this.ValueByIndex index = 
      let mutable value = 0.0
      for i = 0 to this.Indexes.Length - 1 do 
       if this.Indexes.[i] = index then
        value <- berror_buffer.[i]
      value
 
     override this.BackpropError (grad:float array) = 
      Array.init this.Inputs.Length (fun i ->  (this.ValueByIndex i)) 
  
    ///Neural network layer that adds gaussian noise to input values
    type NoiseLayer(ninputs:int,noutputs:int,noiseLevel:float) =
     inherit  NeuralLayer(ninputs,noutputs)
     let inputs =     Array.init 440 (fun i -> Array.init ninputs (fun x -> 0.0))
     let outputs =  Array.init 440 (fun i -> Array.init noutputs (fun x -> 0.0))
     let berror_buffer = Array.init ninputs (fun x -> 0.0)
     let mutable cur_time = 0
     ///Provides access to current outputs array using current timestep
     member this.Outputs = outputs.[cur_time]
     member this.Inputs = inputs.[cur_time]
 
 
 
     //let drops = Array.init 100 (fun x -> Array.init
 
     override this.setTimeStep x = cur_time <- x
     override this.WeightsSize = 0
     override this.setWeight x y = ignore(None)
     override this.getWeight i  = 0.0
     override this.gradSize() = 0
     override this.Size() = noutputs
     override this.GradientNeuron (target:float array) = 
      for i = 0 to target.Length - 1 do
       berror_buffer.[i] <- target.[i]
      [||]
     override this.BackpropError (grad:float array) = 
      Array.init berror_buffer.Length (fun i -> berror_buffer.[i])
 
     override this.ComputeEval() = 
      let proc_inputs = this.Inputs
      for i  = 0 to  proc_inputs.Length - 1 do
        let U1 = rnd_noise.NextDouble()
        let U2 = rnd_noise.NextDouble()
        let nval = noiseLevel *  ((Math.Sqrt (-2.0 * Math.Log U1)) * Math.Cos (Math.PI * 2.0 * U2) - 0.5)
        this.Outputs.[i] <- proc_inputs.[i] + nval
      
  
     override this.Gradient a b c = 
      [||]
  
     override this.LastResult = this.Outputs
 
     override this.setResult  i x = 
      this.Outputs.[i] <- x
 
     override this.SetInput (input:float array) =
      for i = this.LastInputIndex to input.Length - 1 do
       this.Inputs.[i] <- input.[i]
      this.LastInputIndex <-this.LastInputIndex + input.Length
      this.ComputeEval()
  
     override this.SetInputUneval (input:float array) =
      for i =  this.LastInputIndex to input.Length - 1 do
       this.Inputs.[i] <- input.[i]
      this.LastInputIndex <- this.LastInputIndex + input.Length
 
     override this.compute() = 
      this.Outputs  
 
    type CopyLayer(ninputs:int) =
     inherit  NeuralLayer(ninputs,ninputs)
     let noutputs = ninputs
     let inputs =     Array.init 440 (fun i -> Array.init ninputs (fun x -> 0.0))
     let outputs =  Array.init 440 (fun i -> Array.init noutputs (fun x -> 0.0))
     let berror_buffer = Array.init ninputs (fun x -> 0.0)
     let mutable cur_time = 0
     member this.Outputs = outputs.[cur_time]
     member this.Inputs = inputs.[cur_time]
 
 
     override this.setTimeStep x = cur_time <- x
     override this.WeightsSize = 0
     override this.setWeight x y = ignore(None)
     override this.getWeight i  = 0.0
     override this.gradSize() = 0
     override this.Size() = noutputs
 
     override this.GradientNeuron (target:float array) = 
      for i = 0 to target.Length - 1 do
       berror_buffer.[i] <- target.[i]
      [||]
 
     override this.BackpropError (grad:float array) = 
      let p = Array.init berror_buffer.Length (fun i -> berror_buffer.[i])
      p
     // Console.WriteLine ( p |> Array.map (fun x -> x.ToString() + " ") |> Array.fold (+) "")
     // Console.WriteLine()
      //p
   
     override this.ComputeEval() = 
      let proc_inputs = this.Inputs
      for i  = 0 to  proc_inputs.Length - 1 do
       this.Outputs.[i] <- proc_inputs.[i] 
      
  
     override this.Gradient a b c = 
      [||]
  
     override this.LastResult = this.Outputs
 
     override this.setResult  i x = 
      this.Outputs.[i] <- x
 
 
     override this.SetInput (input:float array) =
      for i = this.LastInputIndex to this.LastInputIndex + input.Length - 1 do
       this.Inputs.[i] <- input.[i - this.LastInputIndex]
      this.LastInputIndex <-this.LastInputIndex + input.Length
      this.ComputeEval()
  
     override this.SetInputUneval (input:float array) =
      for i =  this.LastInputIndex to this.LastInputIndex + input.Length - 1 do
       this.Inputs.[i] <- input.[i -  this.LastInputIndex]
      this.LastInputIndex <- this.LastInputIndex + input.Length
 
     override this.compute() = 
      this.Outputs  


 
    [<AbstractClass>] 
    ///Base class for implementing fully connected and other processing layers
    type FullyConnectedLayer(ninputs:int,noutputs:int) =
     inherit  NeuralLayer(ninputs,noutputs) 
    // let weights = Array.init (noutputs*(ninputs+1)) (fun x -> (rnd.NextDouble() * 0.5) - 0.25)        
     let weights = Array.init (noutputs*(ninputs+1)) (fun x -> (rnd.NextDouble() * 0.2) - 0.1)        

     let outputs =  Array.init 110 (fun x -> Array.init noutputs (fun x -> 0.0))
     let proc_inputs = Array.init 110 (fun x -> Array.init (ninputs + 1) (fun x -> 1.0))
     let ngrad = Array.init  (noutputs * (ninputs + 1))  (fun x -> 0.0)
     let null_data = Array.init (noutputs) (fun x -> 0.2)

     let mutable cur_time = 0
     ///Provides access to array of layer weights
     member this.Weights = weights
     member this.Outputs = outputs.[cur_time]
     abstract  member Inputs : unit -> float array with get
     default this.Inputs = proc_inputs.[cur_time]
     member this.curStep = cur_time
     ///Layer outputs at previous time step
     member this.prevOutputs = 
      if cur_time>0 then
       outputs.[cur_time-1]
      else
       null_data
 
     override this.setTimeStep x = cur_time <- x
 
     override this.setWeight i k =
      let mutable p = k
     // if p > 0.5 then p <- 0.5
      //if p < -0.5 then p <- -0.5  
      if not this.Frozen then
       weights.[i] <- p

     override this.getWeight i  =
      weights.[i]
 
     override this.gradSize() = ngrad.Length

     override this.LastResult =  outputs.[cur_time]
 
     override this.SetInput (inputs: float array) =
  
      let pins = this.Inputs
      //Console.WriteLine((this.LastInputIndex.ToString()))
      //Console.WriteLine(inputs.Length)
      for i = this.LastInputIndex  to (inputs.Length - 1) + this.LastInputIndex  do
       pins.[i] <- inputs.[i - this.LastInputIndex]
       //let p =  (i.ToString()) + " " + ((i - (this.LastInputIndex)).ToString())
       //Console.WriteLine(p)
      this.LastInputIndex <- this.LastInputIndex + inputs.Length 
      this.ComputeEval()
      //Console.WriteLine((this.LastInputIndex.ToString()))

     override this.SetInputUneval (inputs: float array) =
      let pins = this.Inputs
      //Console.WriteLine((this.LastInputIndex.ToString()))
      for i =this.LastInputIndex  to (inputs.Length - 1) + this.LastInputIndex  do
       pins.[i] <- inputs.[i - this.LastInputIndex]
      this.LastInputIndex <- this.LastInputIndex + inputs.Length 
  
  
     override this.Size() = noutputs

     override this.setResult i k =
      this.Outputs.[i] <- k
 
     override this.Gradient grad_neuron windex grad_buffer =
       let pins = this.Inputs
       start_profile "Unified general_gradient_compute"

       let mutable i = windex
       for ncount = 0 to grad_neuron.Length - 1  do
        let ng =  grad_neuron.[ncount]
     //   Console.WriteLine("--------")
        for wcount = 0 to  pins.Length - 1 do
         grad_buffer.[i] <- (pins.[wcount]) * ng
     //    Console.WriteLine(grad_buffer.[i])
         i <- i + 1

       end_profile "Unified general_gradient_compute"
       grad_buffer

     override this.BackpropError (grad_neuron:float array) =
      let pins = this.Inputs
      start_profile "unified back error"
      let nsize = pins.Length

      let berror =  Array.init  (nsize - 1)  (fun x -> 0.0)
  
      for j = 0 to grad_neuron.Length - 1 do
       let w =  j * pins.Length
       let g =  grad_neuron.[j]
       for i = 0 to berror.Length - 1 do
         berror.[i] <- berror.[i] + weights.[i+w]  *  g

      end_profile "unified back error"
      berror

    ///Implements softmax output layer with cross-entropy error
    type SoftmaxLayer(ninputs:int,noutputs:int) =
     inherit  FullyConnectedLayer(ninputs,noutputs)
 
     member private  this.renormalize () = 
      let outputs = this.Outputs
      let mutable x = 0
      let t1 =  this.Inputs.Length - 1
      for i =0 to outputs.Length - 1 do
       x <- i * (this.Inputs.Length)
       let mutable kx = 0.0
       for j = x to x + t1 do
        kx <- kx + this.Weights.[j]*this.Weights.[j]
       kx <- Math.Sqrt (kx)
       if kx > 5.0 then
        for j = x to x + t1 do
         this.Weights.[j] <- this.Weights.[j] / (kx*0.2) 
 
 
 
     override this.ComputeEval () =
      let proc_inputs = this.Inputs
      let outputs = this.Outputs
      let weights = this.Weights
      //this.renormalize ()
      start_profile("forward pass softmax")
      //flash outputs

      start_profile("mult")
      //input multiply
      let mutable x = 0
      let t =  proc_inputs.Length
      let t1 =  proc_inputs.Length - 1
      dgemv_cpu weights proc_inputs outputs
    
    //  Native.vect_mat_dgemv  weights proc_inputs outputs
     // for i =0 to outputs.Length - 1 do
      // let x = i * (proc_inputs.Length)
      // let mutable kx = 0.0
       
     
     (*  for j = x to x + t1 do
        kx <- kx + weights.[j] * proc_inputs.[j-x]
        if Double.IsNaN (weights.[j]) then 
          Console.WriteLine("Weight infinity" + (j.ToString()))
     
        if Double.IsNaN  ( proc_inputs.[j-x]) then 
          Console.WriteLine("Input is infiniry" + ((j-x).ToString()))
      
        if Double.IsNaN  (  kx) then 
          Console.WriteLine("KX is infinity" + ((proc_inputs.[j-x]).ToString()))
          
     
       x <- x +  t
       outputs.[i] <- kx*)

      end_profile("mult")

      start_profile("post-comp")
  

      let ymax = Array.max outputs
  
      for i = 0 to outputs.Length - 1 do
       //Console.WriteLine(outputs.[i] / ymax)
  
       outputs.[i] <- Math.Exp (outputs.[i] - ymax)
      let sum_out = Array.sum outputs
   
      for i=0 to outputs.Length - 1 do
       outputs.[i]  <- outputs.[i]  / sum_out
   
      // compute eXes
 
    (* for i = 0 to outputs.Length - 1 do
       let z = outputs.[i]
   
       outputs.[i] <-  Math.Exp (outputs.[i])
       if Double.IsInfinity (outputs.[i]) then
          Console.WriteLine("EX IS infinity " + (z.ToString()) + " " + (outputs.[i].ToString()))
   
       sumx <- sumx + outputs.[i]
      if sumx = 0.0 then 
       Console.WriteLine("ZERO SUM")
      // divide by Ex
      for i = 0 to  outputs.Length - 1  do
        outputs.[i] <- outputs.[i] / sumx
        if Double.IsNaN (outputs.[i]) then
          Console.WriteLine("SOFTMAX NaN " + (sumx.ToString()))
      end_profile("post-comp")*)

      end_profile("forward pass softmax")

     override this.compute() =
      this.Outputs


     override this.GradientNeuron  target_outputs =
      //let p = Array.map2 (fun (x:float) y -> Console.WriteLine((x.ToString()) + " " + (y.ToString()));(x - y)) outputs target_outputs
  
      target_outputs

    let tanh_func = Math.Tanh
    let derivative_tanh = (fun x -> (1.0 - x*x))

    let relU_func = (fun x -> if x > 0.0 then x else 0.0)
    let derivative_relU = (fun x -> if x>0.0 then 1.0 else 0.0)

    let linear_fun = (fun x -> x)
    let linear_derivative= (fun _ -> 1.0)


    let activation_tanh = (tanh_func,derivative_tanh)
    let activation_relU = (relU_func,derivative_relU)
    let activation_linear = (linear_fun,linear_derivative)
  
    ///one dimensional convolutional layer for text processing
    type ConvlolutionalLayer_1D(maxinputs:int,filter_size:int, step_size:int) = 
     inherit  FullyConnectedLayer(maxinputs,(maxinputs/step_size))
     let mutable input_size = maxinputs
     let mutable output_size =  (maxinputs / step_size) 
 
     let mutable activation = tanh_func
     let mutable act_derivative = derivative_tanh
 
 
  
     member private  this.renormalize () = 
      
      let mutable x = 0
      let t1 =  input_size
      for i =0 to  output_size do
       x <- i * (input_size)
       let mutable kx = 0.0
       for j = x to x + t1 do
        kx <- kx + this.Weights.[j]*this.Weights.[j]
       kx <- Math.Sqrt (kx)
       if kx > 1.0 then
        for j = x to x + t1 do
         this.Weights.[j] <- this.Weights.[j] / (kx) 
     
     ///sets activation function for this layer
     member this.setActivation (x,dx) = 
       activation <- x
       act_derivative <- dx
 
     override this.ComputeEval () =
       //this.renormalize ()
       start_profile("forward pass Convlolutional1DLayer")
      // Console.WriteLine("ququ1")
       let proc_inputs = this.Inputs
       let outputs = this.Outputs
       let weights = this.Weights
  
 
       let mutable offset = 0
       let outputs_size =( input_size / step_size) - ((filter_size/ step_size) - 1)
   
       for i = 0 to this.Outputs.Length - 1 do
        outputs.[i] <- 0.0
  
       for i =0 to outputs_size - 1 do
      // let x = i * (proc_inputs.Length)
        let mutable kx = 0.0
        for j = 0 to filter_size - 1 do
         kx <- kx + weights.[j] * proc_inputs.[offset + j]
        outputs.[i] <- kx
        offset <- offset + step_size

      // Console.WriteLine("ququ") 
      // compute results
      // Console.WriteLine(outputs_size)
       for i = 0 to outputs_size - 1 do
        outputs.[i] <- activation (outputs.[i])
        //Console.WriteLine(outputs.[i])
       end_profile("forward pass Convlolutional1DLayer")
       //Console.WriteLine("**")
     override this.compute() = 
      //Console.WriteLine(this.Size())
      this.Outputs
 
     override this.SetInput (inputs: float array) =
      let pins = this.Inputs
     // Console.WriteLine(this.LastInputIndex)
  
      for i = this.LastInputIndex  to inputs.Length - 1 do
       pins.[i] <- inputs.[i]
      this.LastInputIndex <- this.LastInputIndex + inputs.Length
     // Console.WriteLine(pins.[0].ToString())
     // Console.WriteLine(pins.[1].ToString())
      input_size <- this.LastInputIndex
  
      output_size <- this.LastInputIndex / step_size
  
      this.ComputeEval()

     override this.SetInputUneval (inputs: float array) =
      let pins = this.Inputs
      //Console.WriteLine(this.LastInputIndex)
      for i = this.LastInputIndex  to inputs.Length - 1 do
       pins.[i] <- inputs.[i]
  
      this.LastInputIndex <- this.LastInputIndex + inputs.Length
      input_size <- this.LastInputIndex 
      output_size <- this.LastInputIndex  / step_size
   
     override this.WeightsSize = filter_size
     override this.Size() =   (maxinputs / step_size) //output_size
     override this.gradSize() = filter_size
 
     override this.GradientNeuron  target_outputs =
      //Console.WriteLine("grad")
      start_profile "convol GradientNeuron"
      //Console.WriteLine(target_outputs.[0])
    
     // Console.WriteLine ( ((target_outputs.[0..output_size]) |> Array.map (fun x -> x.ToString() + " ") |> Array.fold (+) ""))
      //Console.WriteLine()
  
      let p = Array.map2 (fun (x:float) y -> ((act_derivative x) * y)) (this.Outputs.[0..output_size]) (target_outputs.[0..output_size])
      end_profile "convol GradientNeuron"
     // Console.WriteLine("grad end")
      p
  
     override this.Gradient grad_neuron windex grad_buffer =
       let pins = this.Inputs
       start_profile "softmax general_gradient_compute"

       let mutable i = windex
       for j = 0 to step_size - 1 do 
        grad_buffer.[j+windex] <- 0.0
   
       let mutable offset = 0
       for ncount = 0 to grad_neuron.Length - 1  do
        let ng =  grad_neuron.[ncount]
       // Console.WriteLine("--------")
        i <- windex
        for wcount = 0 to  filter_size - 1 do 
         grad_buffer.[i] <- grad_buffer.[i] + (pins.[wcount + offset]) * ng
         //Console.WriteLine(ng)
         i <- i + 1
        offset <- offset + step_size
   
       for j= windex to (windex + filter_size) - 1 do 
          grad_buffer.[j]  <- grad_buffer.[j]  / ((float) output_size ) 
    
       end_profile "softmax general_gradient_compute"
       grad_buffer
   
     override this.BackpropError (grad_neuron:float array) =
       let pins = this.Inputs
       start_profile "conv-net BackpropError"
       let nsize = pins.Length

       let berror =  Array.init  (nsize - 1)  (fun x -> 0.0)
  
       for j = 0 to grad_neuron.Length - 1 do
        let mutable w = 0
        let g =  grad_neuron.[j]
        for i = 0 to berror.Length - 1 do
          berror.[i] <- berror.[i] + this.Weights.[w]  *  g
          w <- w + 1
          if w > step_size - 1 then
           w <- 0      
      
       end_profile "conv-net BackpropError"
       berror
  
    /// two-dimensional convolutional layer
    type ConvlolutionalLayer_2D(maxinputs:int,input_x:int,input_y:int,filter_x:int, filter_y:int, step_size:int) = 
     inherit  FullyConnectedLayer(maxinputs,(maxinputs/step_size))
     let mutable input_size = maxinputs
     let mutable output_size =  (maxinputs / step_size) 
 
     let mutable activation = tanh_func
     let mutable act_derivative = derivative_tanh
  
     member private this.renormalize () = 
      
      let mutable x = 0
      let t1 =  input_size
      for i =0 to  output_size do
       x <- i * (input_size)
       let mutable kx = 0.0
       for j = x to x + t1 do
        kx <- kx + this.Weights.[j]*this.Weights.[j]
       kx <- Math.Sqrt (kx)
       if kx > 1.0 then
        for j = x to x + t1 do
         this.Weights.[j] <- this.Weights.[j] / (kx) 
     
     
     member this.setActivation (x,dx) = 
       activation <- x
       act_derivative <- dx
 
     override this.ComputeEval () =
       start_profile("forward pass Convlolutional1DLayer")
       let proc_inputs = this.Inputs
       let outputs = this.Outputs
       let weights = this.Weights
     
       let data = convolve proc_inputs weights (input_x,input_y) (filter_x,filter_y) step_size
 
       for i = 0 to data.Length - 1 do
        outputs.[i] <- activation (data.[i])
    
       end_profile("forward pass Convlolutional1DLayer")
 
     override this.compute() = 
      this.Outputs
 
     override this.SetInput (inputs: float array) =
      let pins = this.Inputs 
      for i = this.LastInputIndex  to inputs.Length - 1 do
       pins.[i] <- inputs.[i]
      this.LastInputIndex <- this.LastInputIndex + inputs.Length
      input_size <- this.LastInputIndex
      output_size <- this.LastInputIndex / step_size
  
      this.ComputeEval()

     override this.SetInputUneval (inputs: float array) =
      let pins = this.Inputs
      for i = this.LastInputIndex  to inputs.Length - 1 do
       pins.[i] <- inputs.[i]
      this.LastInputIndex <- this.LastInputIndex + inputs.Length
      input_size <- this.LastInputIndex 
      output_size <- this.LastInputIndex  / step_size
   
     override this.WeightsSize = filter_x * filter_y
     override this.Size() =   (maxinputs / step_size) //output_size
     override this.gradSize() = filter_x * filter_y
 
     override this.GradientNeuron  target_outputs =
      start_profile "convol GradientNeuron"
      let p = Array.map2 (fun (x:float) y -> ((act_derivative x) * y)) (this.Outputs.[0..output_size]) (target_outputs.[0..output_size])
      end_profile "convol GradientNeuron"
      p
  
     override this.Gradient grad_neuron windex grad_buffer =
       let pins = this.Inputs
       start_profile "softmax general_gradient_compute"

       let mutable i = windex
       for j = 0 to step_size - 1 do 
        grad_buffer.[j+windex] <- 0.0
   
       let data = conv_grad pins grad_neuron (input_x,input_y) (filter_x,filter_y) step_size
       for i = windex to windex + data.Length - 1 do
         grad_buffer.[i] <- data.[i-windex]   
  
      //for j= windex to (windex + filter_size) - 1 do 
      //    grad_buffer.[j]  <- grad_buffer.[j]  / ((float) output_size ) 
    
       end_profile "softmax general_gradient_compute"
       grad_buffer
   
     override this.BackpropError (grad_neuron:float array) =
       let pins = this.Inputs
       start_profile "conv-net BackpropError"
       let nsize = pins.Length

       let outputs_size_x =(input_x / step_size) - ((filter_x /step_size) - 1)
       let outputs_size_y =(input_y / step_size) - ((filter_y / step_size) - 1)
   
       let berror = inv_convolve (grad_neuron:float array) (this.Weights) (outputs_size_x,outputs_size_y) (filter_x,filter_y) (step_size)
      
       end_profile "conv-net BackpropError"
       berror



  
    ///Fully connected layer
    type PerceptronLayer(ninputs:int,noutputs:int) =
     inherit  FullyConnectedLayer(ninputs,noutputs)

     let mutable activation = tanh_func
     let mutable act_derivative = derivative_tanh
 
     //let mutable activation =relU_func
     //let mutable act_derivative = derivative_relU
 
 
     let mutable renorm = false
 
     member private this.renormalize () = 
      let outputs = this.Outputs
      let mutable x = 0
      let t1 =  this.Inputs.Length - 1
      for i =0 to outputs.Length - 1 do
       x <- i * (this.Inputs.Length)
       let mutable kx = 0.0
       for j = x to x + t1 do
        kx <- kx + this.Weights.[j]*this.Weights.[j]
       kx <- Math.Sqrt (kx)
       if kx > 10.0 then
        for j = x to x + t1 do
         this.Weights.[j] <- this.Weights.[j] / (kx*0.1 ) 
     
 
 
     member this.setActivation (x,dx) = 
       activation <- x
       act_derivative <- dx
 
     override this.ComputeEval () =

      start_profile("forward pass perceptron")
      let proc_inputs = this.Inputs
      let outputs = this.Outputs
      let weights = this.Weights

     
      //
      //input multiply
      this.renormalize ()

     // forward_pass_cpu weights proc_inputs outputs 
      //forward_pass_mnet weights proc_inputs outputs 
      //Native.vect_mat weights proc_inputs outputs
      dgemv_cpu  weights proc_inputs outputs
     // Native.vect_mat_dgemv  weights proc_inputs outputs
     // forward_pass_opencl weights proc_inputs outputs 
     (* let mutable x = 0
      let t =  proc_inputs.Length
      let t1 =  proc_inputs.Length - 1

      for i =0 to outputs.Length - 1 do
      // let x = i * (proc_inputs.Length)
       let mutable kx = 0.0
       for j = x to x + t1 do
        kx <- kx + weights.[j] * proc_inputs.[j-x]
       x <- x +  t
       outputs.[i] <- kx*)

      // compute results
      for i = 0 to outputs.Length - 1 do
       outputs.[i] <- activation (outputs.[i])
      end_profile("forward pass perceptron")

     override this.compute() =
      this.Outputs

     override this.GradientNeuron  target_outputs =
  
      start_profile "percept GradientNeuron"
      let p = Array.map2 (fun (x:float) y -> ((act_derivative x) * y)) this.Outputs target_outputs
      end_profile "percept GradientNeuron"
      p

  
   ///Recurrent layer for implementing Elman-type RNNs
    type RecurrentLayer(ninputs:int,noutputs:int) =
     inherit  PerceptronLayer(ninputs,noutputs)
     let berror_buffer = Array.init 440 (fun x -> Array.init noutputs (fun x -> 0.0))
 
 
     (*override this.SetInput x =
       let inp = Array.append (x) (this.prevOutputs) 
       base.SetInput inp

     override this.SetInputUneval x = 
      let inp = Array.append (x) (this.prevOutputs) 
      base.SetInput inp*)
     member this.Diagonal() = 
      let mutable x = 0
      let t1 = ninputs - 1
      for i =0 to noutputs - 1 do
      // let x = i * (proc_inputs.Length)
       let mutable kx = 0.0
       for j = x + (ninputs-noutputs) to x + t1 do
        if j-x <> i then this.Weights.[j] <- 0.0 else this.Weights.[j] <- this.Weights.[j] / 2.0

       
     override this.ComputeEval () =
     // Console.WriteLine(noutputs)
     // Console.WriteLine(ninputs)
      for i = (ninputs - noutputs) to ninputs - 1 do
       
       this.Inputs.[i] <- this.prevOutputs.[i - (ninputs - noutputs)]
      base.ComputeEval()
     //override this.Inputs = 
     // let hs = ninputs - noutputs 
 
     // Console.WriteLine ((this.curStep.ToString()) + " data=" + ([|for i in inp -> (i.ToString()) + " "|] |> Array.fold (+) ""))
     // inp  
 
     override  this.GradientNeuron  target_outputs =
      //Console.WriteLine(target_outputs.Length)
      //Console.WriteLine(noutputs)
      let p = base.GradientNeuron [|for i in {0..target_outputs.Length-1} -> target_outputs.[i] + berror_buffer.[this.curStep+1].[i]|]
     // Console.WriteLine ("current berror=" + ([|for i in (berror_buffer.[this.curStep+1]) -> (i.ToString()) + " "|] |> Array.fold (+) ""))
      for i = 0 to berror_buffer.[this.curStep+1].Length - 1 do
         berror_buffer.[this.curStep+1].[i] <- 0.0
      p  
   
     override this.BackpropError (grad_neuron:float array) =
  
      let error = base.BackpropError(grad_neuron)
     // Console.WriteLine ("first berror=" + ([|for i in (error) -> (i.ToString()) + " "|] |> Array.fold (+) ""))
      for i = error.Length - noutputs to error.Length - 1 do
        berror_buffer.[this.curStep].[i - (error.Length - noutputs )] <- error.[i]
      error.[0.. error.Length - noutputs - 1]   

  
  
    type ProjectionLayer (index_size:int,pass_throw_size:int,project_size:int) =
     inherit NeuralLayer(1 + pass_throw_size,project_size + pass_throw_size)
 
     let mutable activation = tanh_func
     let mutable act_derivative = derivative_tanh
 
     //let projections = Array.init index_size (fun x -> (Array.init project_size (fun x -> (rnd.NextDouble() * 0.2) - 0.1)))
 
     let projections = Array.init index_size (fun x -> (Array.init project_size (fun x -> 0.0)))
 
     let outputs = Array.init 450 (fun x -> Array.init (project_size+pass_throw_size) (fun x -> 0.0))
     let inputs = Array.init 450 (fun x -> Array.init (1 + pass_throw_size) (fun x -> 0.0))
     let mutable cur_time = 0
     member this.Outputs = outputs.[cur_time]
     member this.Inputs = inputs.[cur_time]
     member this.Projections = projections 
 
     //let drops = Array.init 100 (fun x -> Array.init
 
     override this.setTimeStep x = cur_time <- x

     member this.eval() =
      // direct out
      start_profile "project - eval"
      let proc_input = this.Inputs
      let output = this.Outputs
  
      let index = (int)(proc_input.[0])
     // Console.WriteLine(index)
      for i = 0 to (project_size-1) do
       output.[i] <- Math.Tanh (projections.[index].[i])
       //output.[i] <-  (projections.[index].[i])
      //pass throw
      for i = project_size to (project_size + pass_throw_size - 1) do
       output.[i] <- proc_input.[i-project_size+1]
      end_profile "project - eval"

     override this.SetInput(inputs_data:float array) =
      start_profile "project - setinput"
      for i = 0 to inputs_data.Length - 1 do
       this.Inputs.[i] <- inputs_data.[i]
      end_profile "project - setinput"
  
      this.eval()

     override this.SetInputUneval(inputs_data:float array) =
      for i = 0 to inputs_data.Length - 1 do
        this.Inputs.[i] <- inputs_data.[i]

     override this.compute() = this.Outputs
     override this.ComputeEval() = 
      this.eval()

  
     override this.Size() = project_size + pass_throw_size
     override this.WeightsSize = index_size*project_size

     override this.getWeight i =
      let x = i/(project_size)
      let y = i-(x*(project_size))
      projections.[x].[y]
  
  
     override this.setWeight i value =
      let x = i/(project_size)
      let y = i-(x*(project_size))
      projections.[x].[y] <- value
      ignore(None)
  
     override this.gradSize() = index_size*project_size
     override this.LastResult = this.Outputs

     override this.setResult i k =
      this.Outputs.[i] <- k

     (*override this.GradientNeuron target_outputs =
       let output = this.Outputs
       let ngrad = Array.init  project_size (fun x -> 0.0)
       for i = 0 to  project_size - 1 do
        ngrad.[i] <-  (1.0 - (output.[i]*output.[i])) * (target_outputs.[i])
       ngrad*)
   
     override this.GradientNeuron  target_outputs =
       start_profile "project - fgrad"
       let p = Array.map2 (fun (x:float) y -> ((act_derivative x) * y)) this.Outputs target_outputs
       end_profile "project - fgrad"
       p

     override this.Gradient grad_neuron windex grad_buffer =
       start_profile "project - fullgrad"
    
      //fill with zeros
       //Console.WriteLine(windex)
       let proc_input = this.Inputs
   
     //  for i = windex to (windex + this.gradSize() - 1) do
     //   grad_buffer.[i] <- 0.0
    
       let index = (int)(proc_input.[0])
       let posit = index * project_size
       //Console.WriteLine(posit)
       for i = posit to (posit + project_size) - 1 do
        grad_buffer.[i+windex] <- grad_neuron.[i-posit] *  projections.[index].[i-posit]
       end_profile "project - fullgrad"
   
       grad_buffer

     override this.BackpropError (grad_neuron:float array) =
      start_profile "project - bpe"
   
      let proc_input = this.Inputs
      let nsize = proc_input.Length

      let berror =  Array.init  (nsize - 1)  (fun x -> 0.0)
      end_profile "project - bpe"
      berror
  