module Lib
    ( someFunc
    ) where
import System.Environment
import Data.Set (Set, fromList, toList, insert, empty)
import Control.Monad.State
import Data.List (intersperse, (\\))

someFunc :: IO ()
someFunc = do
  [f,start,capacity] <- getArgs
  fc <- readFile f 
  let adjlist = parseAdjacencyList fc
  putStrLn $ makeEssenceParam adjlist (read start) (read capacity) 


parseAdjacencyList :: String -> AdjacencyList
parseAdjacencyList f =
  let ls = lines f
      as = filter (\l -> not (head l == '#')) ls
      ws = words <$> as
      is = (read <$>) <$> ws
    in AdjacencyList is

newtype AdjacencyList = AdjacencyList [[Int]]
  deriving Show
  
newtype EdgeSet = EdgeSet (Set (Set Int))
  deriving Show

graphSize :: AdjacencyList -> Int
graphSize (AdjacencyList l) = 1 + maximum (join l)

toEdgeSet :: AdjacencyList -> State EdgeSet ()
toEdgeSet (AdjacencyList []) = return () 
toEdgeSet (AdjacencyList (a:as)) = toEdgeSet' a >> toEdgeSet (AdjacencyList as)
  where
    toEdgeSet' :: [Int] -> State EdgeSet ()
    toEdgeSet' [] = error "toEdgeSet': empty adjacency list!"
    toEdgeSet' [s] = return ()
    toEdgeSet' (f:r) = mapM_ addEdge $ zip (cycle [f]) r
    addEdge :: (Int,Int) -> State EdgeSet ()
    addEdge (l,r) = do
      EdgeSet s <- get
      put $ EdgeSet $ insert (fromList [l,r]) s

toEssenceSet :: EdgeSet -> String
toEssenceSet (EdgeSet es) =
  "{" ++ join (intersperse "," [ toEdge e | e <- toList es]) ++ "}"
  where toEdge :: Set Int -> String
        toEdge s = case toList s of
                     [l,r] -> "{" ++ show l ++ "," ++ show r ++ "}"
                     e -> error $ "toEssenceSet: toEdge: wrong dimension: " ++ show e

makeEssenceParam :: AdjacencyList -> Int -> Int -> String
makeEssenceParam adjlist start capacity =
  unlines [ "letting graph_size be " ++ show (graphSize adjlist)
          , "letting truck_size be " ++ show capacity
          , "letting graph be " ++ graphLit
          , "letting start be record { fire = {" ++ show start ++ "}"
          , "                        , water = {}"
          , "                        , grass = " ++ grassLit 
          , "                        }" 
          ]
  where graphLit = toEssenceSet $ execState (toEdgeSet adjlist) (EdgeSet empty)
        grass = [0..((graphSize adjlist) - 1)] \\ [start]
        grassLit = "{" ++ join (intersperse "," (show <$> grass)) ++ "}"

