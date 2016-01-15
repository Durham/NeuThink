module TestMode
open System

let filter = Array.init 4 (fun i -> 0.0)

let signal = Array.init (32*32) (fun i -> 0.0)



let filter1 = Array.init 4 (fun i -> 0.4)

let signal1 = Array.init (32*32) (fun i -> 1.0)


let signal4 = Array.append (Array.init 32 (fun i -> 0.5)) (Array.init 32 (fun i -> 0.2))
let signal5 = Array.append (Array.init 32 (fun i -> 0.2)) (Array.init 32 (fun i -> 0.2))
let signal6 =  Array.append  signal4  signal5

let filter4 = Array.init 4 (fun i -> 0.4)



let two2one x y size = y*size+x

let convolve (signal:float array) (filter:float array) (signal_x:int,signal_y:int) (filter_x:int,filter_y:int) (stride:int)=
  let outputs_size_x =(signal_x / stride) - ((filter_x / stride) - 1)
  let outputs_size_y =(signal_y / stride) - ((filter_y / stride) - 1)
  let outputs = Array.init (outputs_size_x*outputs_size_y ) (fun x -> 0.0)
  Console.WriteLine (outputs_size_x*outputs_size_y)
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

let rest() = convolve signal6 filter4 (32,4) (2,2) 1

