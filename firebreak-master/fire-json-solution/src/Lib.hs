{-# LANGUAGE OverloadedStrings #-}
module Lib
    ( someFunc
    ) where
import System.Environment
import Data.Text hiding (lines, tail, unlines)
import Shelly
import Text.JSON
import Text.JSON.String
import Data.Either

someFunc :: IO ()
someFunc = do
  [solution] <- getArgs
  j <- shelly $ silently $ run "conjure" ["pretty", "--output-format=json", pack solution]
  let (Right reduced) = runGetJSON readJSObject (unlines $ tail $ lines $ unpack j)
  putStrLn $ (showJSTopType $ forceErr $ reduceJSON reduced) ""


reduceJSON :: JSValue -> Either String JSValue 
reduceJSON v = do
  stmts <- unwrapArray =<< v <?> "mStatements"
  let [seqdecl,dropsdecl] = rights $ (\s -> s <?> "Declaration") <$> stmts
  seqlett  <- unwrapArray =<< seqdecl <?> "Letting"
  let [seqconst] = rights $ (\s -> s <?> "Constant") <$> seqlett 
  seqconstab <- seqconst <?> "ConstantAbstract"
  seqarr <- unwrapArray =<< seqconstab <?> "AbsLitSequence"
  stat <- sequence $ reduceState <$> seqarr
  return $ JSArray stat 

reduceState :: JSValue -> Either String JSValue
reduceState v = do
  cabs <- v <?> "ConstantAbstract"
  reco <- cabs <?> "AbsLitRecord"
  fir <- extractIntSetByKey "fire" reco 
  wat <- extractIntSetByKey "water" reco 
  gra <- extractIntSetByKey "grass" reco 
  return $ encJSDict [("fire" :: String, fir)
                     ,("water" :: String, wat)
                     ,("grass" :: String, gra)
                     ] 

extractIntSetByKey :: String -> JSValue -> Either String JSValue
extractIntSetByKey key (JSArray v) = do
  vs <- sequence $ unwrapArray <$> v
  case Prelude.filter (elem (encJSDict [("Name" :: String,showJSON $ toJSString key)])) vs of
    [fireset] -> do
      let [recarr] = rights $ (\s -> s <?> "ConstantAbstract") <$> fireset 
      setarr <- recarr <?> "AbsLitSet" 
      extractIntSet setarr
    _ -> Left $ "extractIntSetByKey: " ++ key ++ ": problem extracting key"
extractIntSetByKey k _ = Left $ "extractIntSetByKey: " ++ k ++ ": expected JSArray"

extractIntSet :: JSValue -> Either String JSValue
extractIntSet v = do
  arr <- unwrapArray v
  intarr <- sequence $ extractConstantInt <$> arr
  return $ JSArray intarr 
  where
    extractConstantInt :: JSValue -> Either String JSValue
    extractConstantInt v = do
      x <- unwrapArray =<< v <?> "ConstantInt"
      case x of
        [_,i] -> return i
        _ -> Left "malformed constant int"


forceErr :: Either String a -> a
forceErr (Right a) = a
forceErr (Left er) = error er

unwrapArray :: JSValue -> Either String [JSValue]
unwrapArray (JSArray a) = Right a
unwrapArray e = Left $ "not a JSON array" ++ "\n" ++ show e

(<?>) :: JSValue -> String -> Either String JSValue
(<?>) (JSObject o) k =
  case lookup k $ fromJSObject o of
    Nothing -> Left $ "key not found: " ++ k ++ "\n" ++ showJSObject o "" 
    Just so -> Right so
(<?>) e _ = Left $ "not a JSON object" ++ "\n" ++ show e 
