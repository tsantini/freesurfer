project(AANsegment)

host_os()

# FIX ME: Currently cannot run linux binary on older CentOS7, Ubuntu18
if(APPLE OR HOST_OS MATCHES "CentOS8stream" OR HOST_OS MATCHES "CentOS8" OR HOST_OS MATCHES "Rocky8" OR HOST_OS MATCHES "Ubuntu20" OR HOST_OS MATCHES "Ubuntu22")
   if(APPLE)
     set(BIN_PROGRAMS
       mac_osx/run_segmentNuclei.sh
     )
     install_osx_app(mac_osx/segmentNuclei.app)
   else()
     set(BIN_PROGRAMS
       linux_x86_64/segmentNuclei
       linux_x86_64/run_segmentNuclei.sh
     )
   endif()

   install_symlinks(
     ${BIN_PROGRAMS}
     SegmentAAN.sh
     TYPE programs
     DESTINATION bin
   )

   install_symlinks(
     targetReg.mgz
     targetWorkingres.mgz
     AtlasMesh.gz
     compressionLookupTable.txt
     freeview.lut.txt
     TYPE files
     DESTINATION average/AAN/atlas
   )
else()
   message(WARNING "AANsegment install not supported on HOST_OS=${HOST_OS}")
endif()

