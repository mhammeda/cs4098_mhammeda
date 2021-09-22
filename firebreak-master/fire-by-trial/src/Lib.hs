{-# LANGUAGE OverloadedStrings #-}
module Lib
    ( someFunc
    ) where
import System.Environment
import Shelly
import Data.Text (pack,unpack,split)
import qualified Data.Text as T
import Control.Exception
import Data.List.Split
import System.Directory

someFunc :: IO ()
someFunc = do
  [model_file,trial_file,graph_dir,solution_dir,timeout_seconds] <- getArgs
  trials <- readTrials trial_file
  mapM_ (runTrial graph_dir model_file solution_dir (read timeout_seconds)) trials


   
doInitialModelling :: String -> IO ()
doInitialModelling model_file = do
  putStrLn "Doing initial modelling"
  shelly $ run_ "conjure" ["modelling", "-ac", pack model_file]

removeConjureOutput :: IO ()
removeConjureOutput = do
  putStrLn "Removing conjure-output"
  shelly $ run_ "rm" ["-r","conjure-output"]


reapOrphans :: String -> IO ()
reapOrphans problem = do
  ps <- shelly $ silently $ run "ps" []
  let saviles = filter (T.isInfixOf (pack problem)) $ filter (T.isInfixOf "savilerow") $ T.lines ps
      minions = filter (T.isInfixOf (pack problem)) $ filter (T.isInfixOf "minion") $ T.lines ps
      todie = saviles ++ minions
  if length todie > 0
     then do
       mapM_ reap todie
     else return ()
  where
    reap :: T.Text -> IO ()
    reap t = do
      let pid = head $ T.words t
      putStrLn $ "Reaping " ++ unpack t
      shelly $ run_ "kill" ["-9",pid]


  

data Trial =
  Trial {
    adjlist_file   :: String 
  , set_alight     :: Int
  , water_capacity :: Int
  } deriving Show


(<///>) :: String -> String -> String
(<///>) l r = unpack $ toTextIgnore $ fromText (pack l) </> fromText (pack r)

readTrials :: String -> IO [Trial]
readTrials trial_file = do
  fc <- readFile trial_file 
  return $ readTrial <$> lines fc
  where
    readTrial :: String -> Trial
    readTrial l = case words l of
                    [f,s,w] -> Trial f (read s) (read w)  

runTrial :: String -> String -> String -> Int ->  Trial -> IO ()
runTrial graph_dir model_file solution_dir timeout_seconds trial = do
  putStrLn ""
  let graph_path = pack $ graph_dir <///> adjlist_file trial
  param <- shelly $ silently $ run "fire-runner" [ graph_path 
                                                 , pack $ show $ set_alight trial
                                                 , pack $ show $ water_capacity trial
                                                 ]  
  let trial_name = adjlist_file trial ++ "_" ++ show (set_alight trial) ++ "_" ++ show (water_capacity trial)
  putStrLn $ "Writing parameter file " ++ (trial_name ++ ".param")
  writeFile (trial_name ++ ".param") $ unpack param
  shelly $ catchany_sh (trySolveWithTimeout graph_dir (adjlist_file trial) model_file trial_name solution_dir timeout_seconds)
                       (handleTimeout solution_dir trial_name timeout_seconds)
  putStrLn $ "Moving parameter file to results"
  shelly $ run_ "mv" [pack (trial_name ++ ".param"), pack solution_dir]


trialSolved :: String -> String -> String-> IO Bool
trialSolved model_file solution_dir trial_name = do
  sols <- listDirectory solution_dir
  return $ (trial_name ++ ".solution") `elem` sols 

trySolveWithTimeout :: String -> String -> String -> String -> String -> Int -> Sh ()
trySolveWithTimeout graph_dir adjlist_file model_file trial_name solution_dir timeout_seconds = do
  solved <- liftIO $ trialSolved model_file solution_dir trial_name
  if solved
    then liftIO $ putStrLn "Trial solved already"
    else do
      liftIO $ putStrLn $ "Solving with " ++ show timeout_seconds ++ "s timeout"
      run_ "conjure" ["solve", "-ac", pack model_file, pack (trial_name ++ ".param"), "--limit-time=" <> pack (show timeout_seconds)] 
      liftIO $ putStrLn $ "Moving solution file to " ++ solution_dir
      run_ "mv" [pack (modelPrefix model_file ++ "-" ++ trial_name ++ ".solution"), pack (solution_dir <///> (trial_name ++ ".solution"))]
      liftIO $ putStrLn $ "Making visualisation"
      html <- silently $ run "fire-visulizer" [pack (graph_dir <///> adjlist_file), pack (solution_dir <///> (trial_name ++ ".solution"))]
      liftIO $ writeFile (solution_dir <///> (trial_name ++ ".html")) (unpack html)  

modelPrefix :: String -> String
modelPrefix s = head $ splitOn "." s

handleTimeout :: String -> String -> Int -> (SomeException -> Sh ())
handleTimeout solution_dir trial_name timeout_seconds = \exception -> do
  ex <- lastExitCode
  liftIO $ reapOrphans trial_name
  if ex == 0
    then do
      liftIO $ putStrLn $ "Program killed"
      throw exception
    else do
      laststderr <- lastStderr
      liftIO $ putStrLn $ "Reason for failure written to " ++ (solution_dir <///> (trial_name ++ ".failure"))
      liftIO $ writeFile (solution_dir <///> (trial_name ++ ".failure")) (unpack laststderr) --(show ex)


