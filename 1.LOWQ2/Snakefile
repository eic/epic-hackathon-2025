
# Rule to generate root file with Tagger Tracker tensors using eicrecon
rule taggertracker_training_tensors:
    output:
        "{outputdir}/tensors_{run_number}.eicrecon.tree.edm4eic.root",
    params:
        input=lambda wildcards: expand("root://dtn-eic.jlab.org//work/eic2/EPIC/RECO/24.12.0/epic_craterlake/SIDIS/pythia6-eic/1.0.0/10x100/q2_0to1/pythia_ep_noradcor_10x100_q2_0.000000001_1.0_run{run_number}.ab.{number:04d}.eicrecon.tree.edm4eic.root", run_number=wildcards.run_number, number=range(0, 100) ),
        compact_xml="epic_lowq2.xml",
    shell:  
        """
        eicrecon {params.input} -Ppodio:output_file={output} \
        -Ppodio:output_collections=TaggerTrackerFeatureTensor,TaggerTrackerTargetTensor \
        -Pplugins_to_ignore=janatop,LUMISPECCAL,ECTOF,BTOF,FOFFMTRK,RPOTS,B0TRK,MPGD,ECTRK,DRICH,DIRC,pid,tracking,acts,EEMC,BEMC,FEMC,EHCAL,BHCAL,FHCAL,B0ECAL,ZDC,BTRK,BVTX,PFRICH,richgeo,evaluator,pid_lut,reco,rootfile \
        -Pdd4hep:xml_files={params.compact_xml} \
        """

rule split_tensors_features:
     output:
        "features_lowq2.edm4eic.root",
     input:
        "test_tensors.eicrecon.tree.edm4eic.root",
     params:
        compact_xml="epic_lowq2.xml",     
     shell:
        """
        eicrecon {input} -Ppodio:output_file={output} \
        -Pjana:nevents=100000 \
        -Ppodio:output_collections=TaggerTrackerFeatureTensor,TaggerTrackerTargetTensor \
        -Pplugins_to_ignore=janatop,LUMISPECCAL,ECTOF,BTOF,FOFFMTRK,RPOTS,B0TRK,MPGD,ECTRK,DRICH,DIRC,pid,tracking,acts,EEMC,BEMC,FEMC,EHCAL,BHCAL,FHCAL,B0ECAL,ZDC,BTRK,BVTX,PFRICH,richgeo,evaluator,pid_lut,reco,rootfile \
        -Pdd4hep:xml_files={params.compact_xml} \
        """
	
rule split_tensors_targets:
     output:
        "targets_lowq2.edm4eic.root",
     input:
        "test_tensors.eicrecon.tree.edm4eic.root",
     params:
        compact_xml="epic_lowq2.xml",
     shell:
        """
        eicrecon {input} -Ppodio:output_file={output} \
        -Pjana:nevents=100000 \
        -Ppodio:output_collections=TaggerTrackerTargetTensor \
        -Pplugins_to_ignore=janatop,LUMISPECCAL,ECTOF,BTOF,FOFFMTRK,RPOTS,B0TRK,MPGD,ECTRK,DRICH,DIRC,pid,tracking,acts,EEMC,BEMC,FEMC,EHCAL,BHCAL,FHCAL,B0ECAL,ZDC,BTRK,BVTX,PFRICH,richgeo,evaluator,pid_lut,reco,rootfile \
        -Pdd4hep:xml_files={params.compact_xml} \
        """
	
rule split_tensors_result:
     output:
        "submit_lowq2.edm4eic.root",
     input:
        "test_tensors.eicrecon.tree.edm4eic.root",
     params:
        compact_xml="epic_lowq2.xml",
     shell:
        """
        eicrecon {input} -Ppodio:output_file={output} \
        -Pjana:nevents=100000 \
        -Ppodio:output_collections=TaggerTrackerPredictionTensor \
        -Pplugins_to_ignore=janatop,LUMISPECCAL,ECTOF,BTOF,FOFFMTRK,RPOTS,B0TRK,MPGD,ECTRK,DRICH,DIRC,pid,tracking,acts,EEMC,BEMC,FEMC,EHCAL,BHCAL,FHCAL,B0ECAL,ZDC,BTRK,BVTX,PFRICH,richgeo,evaluator,pid_lut,reco,rootfile \
        -Pdd4hep:xml_files={params.compact_xml} \
        """