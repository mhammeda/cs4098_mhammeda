{-# LANGUAGE OverloadedStrings #-}
module Lib
    ( someFunc
    ) where
import System.Environment
import Shelly
import Data.Text (pack, unpack)

someFunc :: IO ()
someFunc = do
  [adjlist,solution] <- getArgs
  g <- shelly $ silently $ run "fire-graph-json" [pack adjlist] 
  s <- shelly $ silently $ run "fire-json-solution" [pack solution]
  putStrLn $ makeHtml (filter (/= '\"') $ unpack g) (filter (/= '\"') $ unpack s)
  

makeHtml :: String -> String -> String
makeHtml gData firespread = unlines
  ["<head>"
  ,"  <style> body { margin: 0; } </style>"
  ,"  <script src=\"https://unpkg.com/3d-force-graph\"></script>"
  ,"</head>"
  ,""
  ,"<body>"
  ,"   <button onclick=\"startFire()\">Go</button>" 
  ,"   <button onclick=\"resetFire()\">reset</button>"
  ,"   <div id=\"3d-graph\"></div>"
  ,""
  ,"  <script>"
  ,"    const gData = " ++ gData
  ,""
  ,"    x = 0"
  ,"    const firespread =" ++ firespread
  ,"    var Graph = ForceGraph3D()"
  ,"      (document.getElementById('3d-graph')).graphData(gData);"
  ,"      Graph.nodeColor(function(d) {  if (firespread[x].fire.includes(d.id) ) {" 
  ,"                                         return 'red'"
  ,"                                       } "
  ,"                                       if (firespread[x].water.includes(d.id) ) {"
  ,"                                         return 'blue'"
  ,"                                       }"
  ,"                                       return 'green'"
  ,"                                    } );"
  ,"    var fireInterval;"
  ,"    function startFire() {"
  ,"      resetFire();"
  ,"      fireInterval = setInterval(function() {"
  ,"        if ( x == firespread.length - 1) {"
  ,"          clearinterval(fireinterval);"
  ,"          return;"
  ,"        }"
  ,"        Graph.nodeColor(function(d) {  if (firespread[x].fire.includes(d.id) ) {" 
  ,"                                         return 'red'"
  ,"                                       } "
  ,"                                       if (firespread[x].water.includes(d.id) ) {"
  ,"                                         return 'blue'"
  ,"                                       }"
  ,"                                       return 'green'"
  ,"                                    } );"
  ,"        x = (x + 1) % firespread.length;"
  ,""
  ,"      },1000)"
  ,"    }"
  ,"    function resetFire() {"
  ,"      clearInterval(fireInterval);"
  ,"      x = 0;"
  ,"      Graph.nodeColor(function(d) {  if (firespread[x].fire.includes(d.id) ) {" 
  ,"                                         return 'red'"
  ,"                                       } "
  ,"                                       if (firespread[x].water.includes(d.id) ) {"
  ,"                                         return 'blue'"
  ,"                                       }"
  ,"                                       return 'green'"
  ,"                                    } );"
  ,"    }"
  ,""
  ,"  </script>"
  ,"</body>"
  ]
