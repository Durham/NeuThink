namespace NeuThink
module DataSources =

 type IInputProvider =
   // abstract method
   abstract member Item: int -> float array  with get
   abstract member Length: int with get
   abstract member GetSlice : int option *int option ->  IInputProvider

 type IOutputProvider =
   // abstract method
   abstract member Item: int -> float array  with get
   abstract member Length: int with get
   abstract member GetSlice : int option *int option -> IOutputProvider

 ///A simple in-memory data provider type for training neural net models
 type SimpleProvider(data :float array array) =
  member this.Length with get() = data.Length
  member this.Item with get(i) = data.[i]
 
  interface IInputProvider with
   member this.Item with get(i) = data.[i]
   member this.Length with get() = data.Length
   member this.GetSlice (x,y) =
    let s1, f1 = defaultArg x 0, defaultArg y (data.Length)
    let p = new SimpleProvider(data.[s1..f1])
    p  :> IInputProvider


  interface IOutputProvider with
   member this.Item with get(i) = data.[i]
   member this.Length with get() = data.Length
   member this.GetSlice (x,y) =
    let s1, f1 = defaultArg x 0, defaultArg y (data.Length)
    (new SimpleProvider(data.[s1..f1] ) )  :> IOutputProvider


