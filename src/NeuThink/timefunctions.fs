namespace  NeuThink
open System
open System.Diagnostics
open System.Collections.Generic

module timeFunction =



    let timeProfile = new Dictionary<string,int>()
    let keyProfile = new Dictionary<string,Stopwatch>()

    let mutable st = new Stopwatch()

    let show_time_profile () = 
     for t in keyProfile do
      Console.WriteLine (t.Key + " = " + ((t.Value.ElapsedMilliseconds).ToString()))

    let start_profile key =
     if (keyProfile.ContainsKey (key)) then
      (keyProfile.[key]).Start()
     else
      keyProfile.[key] <- new Stopwatch()
      (keyProfile.[key]).Start()

    let reset_key key =
       keyProfile.[key] <- new Stopwatch()

    let end_profile key =
      (keyProfile.[key]).Stop()
      if timeProfile.ContainsKey(key) then
       timeProfile.[key] <- timeProfile.[key] + (int)(keyProfile.[key]).ElapsedMilliseconds
      else
       timeProfile.[key] <-  (int)(keyProfile.[key]).ElapsedMilliseconds

    let () =
     timeProfile.["features_generation"] <- 0
     timeProfile.["word_shapes"] <- 0
     timeProfile.["features_transform"] <- 0

 
 
