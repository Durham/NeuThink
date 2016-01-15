module DataAccess
open System
open System.IO
open System.Text.RegularExpressions
open System.Collections.Generic


type manufacturers() =
 let manufacts = new ResizeArray<string>()
 let nomanufacts = new ResizeArray<string>()
 member this.Initialize() =
  let data = File.ReadAllLines("manufacts.txt")
  for d in data do
   manufacts.Add(d)
  let data1 = File.ReadAllLines("nomanufacts.txt")
  for d in data1 do
   nomanufacts.Add(d)
 member this.is_known_manufact str =
  Array.ofSeq manufacts |> Array.exists (fun x -> x = str) 
 member this.is_known_nomanufact str =
  Array.ofSeq nomanufacts |> Array.exists (fun x -> x = str) 
 member this.AddManufact manuf =
  if (not (this.is_known_manufact manuf)) && (not (this.is_known_nomanufact manuf)) then
   manufacts.Add(manuf)
 member this.Store() =
  let lines = Array.ofSeq manufacts
  File.WriteAllLines("manufacts.txt",lines)
  
type entity_classes() =
 let manufacts = new ResizeArray<string>()
 let nomanufacts = new ResizeArray<string>()
 member this.Initialize() =
  let data = File.ReadAllLines("entity_classes.txt")
  for d in data do
   manufacts.Add(d)
  let data1 = File.ReadAllLines("no_entity_classes.txt")
  for d in data1 do
   nomanufacts.Add(d)
 member this.is_known (str:string) =
  Array.ofSeq manufacts |> Array.exists (fun x -> x.ToUpper() = str.ToUpper()) 
 member this.is_known_not (str:string) =
  Array.ofSeq nomanufacts |> Array.exists (fun x -> x.ToUpper() = str.ToUpper()) 
 member this.Add manuf =
  if (not (this.is_known manuf)) && (not (this.is_known_not manuf)) then
   manufacts.Add(manuf)
 member this.Store() =
  let lines = Array.ofSeq manufacts
  File.WriteAllLines("entity_classes.txt",lines)

  
  
type structured_enitity_provider() = 
 let entities = new ResizeArray<Dictionary<string,string>>()
 member this.AddEntry (entry:Dictionary<string,string>) =
   entry.["UID"] <- (Guid.NewGuid()).ToString()
   entities.Add(entry)
   
 member this.Entries with get() = Array.ofSeq entities 
 member this.EntriesDirect =  entities
 member this.getEntry i = entities.[i]
 
 
 member this.entry2string (entry:Dictionary<string,string>) =
   let r = new ResizeArray<string>()
   for d in entry do
    r.Add(d.Key + " = " + d.Value)
   (String.concat " \n" (Array.ofSeq r)) + "\n" 
 
 member this.entry2string_serial (entry:Dictionary<string,string>) =
  entry.["word"] + " " + entry.["targetp"]  + " " + entry.["target"]

 member this.save_serial_output filename = 
  File.WriteAllLines(filename,[|for x in this.Entries -> this.entry2string_serial x|])

 member this.save_multi_output filename =
   File.WriteAllLines(filename, [|for x in this.Entries -> this.entry2string x|])
 
 member this.load_multi_output filename =
   let lines = File.ReadAllLines(filename)
   let mutable state = 0
   let mutable dict = new Dictionary<string,string>()
   for line in lines do
    if (line.Length > 2 ) then
     let data = line.Split([|" = "|],StringSplitOptions.RemoveEmptyEntries)
     dict.[data.[0].Trim()] <- (data.[1]).Trim()
    if (line.Length < 2) then
     this.AddEntry(dict)
     dict <-  new Dictionary<string,string>()
 
 member this.getClasses(class_field_name:string) =
    if (this.Entries.[0]).ContainsKey(class_field_name) then
      let data = this.Entries
      let classes = (set (data |> Array.map (fun x -> x.[class_field_name])))
      Array.ofSeq classes
    else
      [||]

 member this.EntriesStr i = this.entry2string (this.Entries.[i])

 member this.load_serial_input filename =
  let data = File.ReadAllLines(filename) |>  Array.filter (fun x -> x.Length>0) |>  Array.map (fun x -> if x.Contains(" ") then x else x + " OTHER")
  entities.Clear()
  let cnv = [for x in data -> let p = x.Split([|" "|],StringSplitOptions.RemoveEmptyEntries) in if p.Length > 1 then (p.[0],(if p.[1] <>"OTHER" then "explicit" else "OTHER")) else ("","OTHER")]
  //let cnv = [for x in data -> let p = x.Split([|" "|],StringSplitOptions.RemoveEmptyEntries) in if p.Length > 2 then (p.[0],(if p.[1] <>"OTHER" then p.[2] else "OTHER")) else ("","OTHER")]

  for el in cnv do
   let d = new Dictionary<string,string>()
   d.Add("word",fst el)
   d.Add("target",snd el)
   d.Add("targetp",snd el)
   if (snd el).Contains("42") then
    Console.WriteLine("pufpuf")
    Console.WriteLine(snd el)
    Console.WriteLine(filename)

  (* if snd el = "0" then
    d.Add("target","0")
    d.Add("targetp","0")
   else
    d.Add("target","1")
    d.Add("targetp","1")*)
   this.AddEntry d

 member this.load_serial_input_unf filename =
  let data = File.ReadAllLines(filename) |>  Array.filter (fun x -> x.Length>0) |>  Array.map (fun x -> if x.Contains(" ") then x else x + " OTHER")
  entities.Clear()
  let cnv = [for x in data -> let p = x.Split([|" "|],StringSplitOptions.RemoveEmptyEntries) in if p.Length > 1 then (p.[0],p.[1]) else (p.[0],"OTHER")]
  for el in cnv do
   let d = new Dictionary<string,string>()
   d.Add("word",fst el)
   d.Add("target",snd el)
   d.Add("targetp",snd el)
   if (snd el).Contains("42") then
    Console.WriteLine("pufpuf")
    Console.WriteLine(snd el)
    Console.WriteLine(filename)

  (* if snd el = "0" then
    d.Add("target","0")
    d.Add("targetp","0")
   else
    d.Add("target","1")
    d.Add("targetp","1")*)
   this.AddEntry d
 

//job data providers
type jobdataReader(filename:string) =
 let parseJob (job:string) = 
  let components = job.Split([|";"|],StringSplitOptions.RemoveEmptyEntries)
  let job_dict = new Dictionary<string,string>()
  job_dict.["id"] <- components.[0]
  job_dict.["company_name"] <- components.[1]
  job_dict.["job_type"] <- components.[2]
  job_dict.["job_input"] <- components.[3]
  job_dict.["job_output"] <- components.[4]
  job_dict.["job_datasource"] <- components.[5]
  job_dict
  
 member  this.loadJobsFile () =
  File.ReadAllLines(filename) |> Array.filter (fun x -> x.[0] <> '#') |> Array.map parseJob

 member this.storeJobdata2 (jobdata:structured_enitity_provider) job_output_type job_id = 
  match job_output_type with
  | "fileOutput" -> let lines = jobdata.Entries |> Array.mapi (fun i x -> jobdata.EntriesStr i) 
                    File.WriteAllLines (job_id,lines)
  | _ -> ignore(None)



//some important global instances

let aux_entities = new structured_enitity_provider()
   
