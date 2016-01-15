open System
open System.IO

let vdata = File.ReadAllLines("word_projections-80.txt") |> Array.map (fun x-> x.Split([|" ";"\t"|],StringSplitOptions.RemoveEmptyEntries))

let words = vdata  |> Array.map (fun x -> x.[0])
let vectors = vdata  |> Array.map (fun x -> (x.[1..]) |> String.concat " ")

let() =
 File.WriteAllLines("features.dict",words)
 File.WriteAllLines("features.vect",vectors)

