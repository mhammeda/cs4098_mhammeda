 #!/bin/bash
 ulimit -v hard # virtual memory                                                            
 ulimit -t hard # CPU usage                                                                 
 ulimit -u hard # maximum number of processes
 nice ./gnu_arguments/commands_host.sh