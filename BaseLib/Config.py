##############################################################
# Config.spy
#
# Ini configuration file interface class
#
# License:  MIT 2019 Itai Danielli
##############################################################
import os
import configparser
from BaseLib import Files





class ConfigCls:
    # Class Constructor
    def __init__(self, file_name:str):
        self.file_name = file_name
        self.config = configparser.ConfigParser()
        self.config.read(file_name)

    def Save(self):
        self.config.write(self.file_name)





    #Data download settings.
    def DownloadUrl(self)->str:
        url:str="";
        if(None != self.config):url = self.config['Download']['url']
        return(url);

    def DownloadDir(self)->str:
        dir:str="";
        if(None != self.config):dir = self.config['Download']['dir']
        return(dir);

    def DownloadFile(self)->str:
        file:str="";
        if(None != self.config):file = self.config['Download']['file']
        return(file);

    def DownloadSize(self)->int:
        size_bytes:int="";
        if(None != self.config):size_bytes = self.config['Download']['size_bytes']
        return(int(size_bytes));





    #Tenzorflow session settings.
    def SessionVocSizeGet(self)->int:
        voc_size:str="";
        if(None != self.config):voc_size = self.config['Session']['voc_size']
        return(int(voc_size));

    def SessionStepsGet(self)->int:
        num_steps:str="";
        if(None != self.config):num_steps = self.config['Session']['num_steps']
        return(int(num_steps));





    #Model settings.
    def ModelBatchSizeSet(self, size)->int:
        self.config['Model']['batch_size'] = str(size)
        return;
    def ModelBatchSizeGet(self)->int:
        batch_size:str="";
        if(None != self.config):batch_size = self.config['Model']['batch_size']
        return(int(batch_size));

    # Dimension of the embedding vector.
    def ModelEmbedSizeSet(self, size:int)->int:
        self.config['Model']['embedding_size'] = str(size);
        return;
    def ModelEmbedSizeGet(self)->int:
        embedding_size:str="";
        if(None != self.config):embedding_size = self.config['Model']['embedding_size']
        return(int(embedding_size));

    # How many words to consider left and right.
    def ModelSkipWindowGet(self,window:int)->int:
        self.config['Model']['skip_window'] = str(window)
        return;
    def ModelSkipWindowGet(self)->int:
        skip_window:str="";
        if(None != self.config):skip_window = self.config['Model']['skip_window']
        return(int(skip_window));

    # How many times to reuse an input to generate a label.
    def ModelNumSkipsSet(self, num:int)->int:
        self.config['Model']['num_skips'] = str(num);
        return;
    def ModelNumSkipsGet(self)->int:
        num_skips:str="";
        if(None != self.config):num_skips = self.config['Model']['num_skips']
        return(int(num_skips));

    # Number of negative examples to sample.
    def ModelNumSampledSet(self, num:int)->int:
        self.config['Model']['num_sampled']=str(num)
        return
    def ModelNumSampledGet(self)->int:
        num_sampled:str="";
        if(None != self.config):num_sampled = self.config['Model']['num_sampled']
        return(int(num_sampled));





    #Validation settings.
    # Random set of words to evaluate similarity on.
    def ValidationSize(self)->int:
        valid_size:str="";
        if(None != self.config): valid_size = self.config['Validation']['valid_size']
        return(int(valid_size));

    # Only pick dev samples in the head of the distribution.
    def ValidationWindow(self)->int:
        valid_window:str="";
        if(None != self.config): valid_window = self.config['Validation']['valid_window']
        return(int(valid_window));





    #Report settings.
    def RepLossStep(self)->int:
        loss_step:str="";
        if(None != self.config): loss_step = self.config['Reporting']['loss_step']
        return(int(loss_step));

    def RepSimStep(self)->int:
        sim_eval_ste:str="";
        if(None != self.config): sim_eval_ste = self.config['Reporting']['sim_eval_step']
        return(int(sim_eval_ste));




    #Output settings.
    def OutDirSet(self, dir:str)->str:
        self.config['Output']['out_dir'] = dir
        return;

    def OutDirGet(self)->str:
        out_dir:str="";
        if(None != self.config): out_dir = self.config['Output']['out_dir']
        return(out_dir);

    def OutLogFile(self)->str:
        log_file:str="";
        if(None != self.config): log_file = self.config['Output']['log_file']
        log_file = os.path.join(self.OutDirGet(), log_file)
        return(log_file);

    def OutMetaFile(self) -> str:
        meta_file: str = "";
        if (None != self.config): meta_file = self.config['Output']['meta_file']
        meta_file = os.path.join(self.OutDirGet(), meta_file)
        return (meta_file);

    def OutModelFile(self) -> str:
        model_file: str = "";
        if (None != self.config): model_file = self.config['Output']['model_file']
        model_file = os.path.join(self.OutDirGet(), model_file)
        return (model_file);

    def OutPlotFile(self) -> str:
        plot_file: str = "";
        if (None != self.config): plot_file = self.config['Output']['plot_file']
        plot_file = os.path.join(self.OutDirGet(), plot_file)
        return (plot_file);







