cmd_Release/cream.node := ln -f "Release/obj.target/cream.node" "Release/cream.node" 2>/dev/null || (rm -rf "Release/cream.node" && cp -af "Release/obj.target/cream.node" "Release/cream.node")
