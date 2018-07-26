package beer.data.judgments

import java.io.File
import scala.io.Source
import scala.collection.mutable.{Map => MutableMap}

private class WMT11 (dir:String) {
  
  val csv = dir+"/wmt11-maneval-indivsystems.RNK_results.csv"

  val references_dir = dir+"/wmt11-data/plain/references/" // newstest2011-ref.$lang
  val system_dir = dir+"/wmt11-data/plain/system-outputs/newstest2011/" //  lang pairs dirs
  val langs = List("cs", "de", "es", "fr", "en")
  val lang_pairs = List("cs-en", "de-en", "es-en", "fr-en", "en-cs", "en-de", "en-es", "en-fr")
  
  type LangPair = String
  type Lang = String
  type System = String
  
  val system_sents = MutableMap[LangPair, MutableMap[System, Array[String]]]()
  val ref_sents = MutableMap[LangPair, Array[String]]()
  var judgments = List[Judgment]()

  def load() : Unit = {
    load_system_translations();
    load_references();

    load_csv("wmt11", csv);
  }
  
  private def load_csv(dataset_name:String, csv_fn:String) : Unit = {
    var line_id = 0
    val file_iterator = Source.fromFile(csv_fn).getLines()
    file_iterator.next() // skip_first line
    for(line <- file_iterator){
      val fields = line.split(",")
      
      val src_lang_long = fields(0)
      val tgt_lang_long = fields(1)
      val src_lang_short = WMT11.long_to_short(src_lang_long)
      val tgt_lang_short = WMT11.long_to_short(tgt_lang_long)
      val sys_names = List(fields(7), fields(9), fields(11), fields(13), fields(15))
      val rankings = List(fields(16).toInt, fields(17).toInt, fields(18).toInt, fields(19).toInt, fields(20).toInt);
      val sentId = fields(2).toInt-1
      val lp = s"$src_lang_short-$tgt_lang_short"
      val sents : List[String] = sys_names.map{ sys =>
        if(sys equals "_ref"){
          ref_sents(lp)(sentId)
        }else{
          val long_sys_name = "newstest2011."+(lp.replace("cs", "cz"))+"."+(sys.replace("uedin","udein"))
          system_sents(lp)(long_sys_name)(sentId)
        }
      }
      val ref : String = ref_sents(lp)(sentId)
      
      judgments ::= new Judgment(dataset_name, src_lang_short, tgt_lang_short, sentId, sys_names, rankings, sents, ref)
    }
  }
  
  private def load_system_translations() : Unit = {
    for(lp <- lang_pairs){
      val system_sents = scala.collection.mutable.Map[String, Array[String]]()
      val lp_real = lp.replace("cs","cz")
      for(file <- WMT11.getListOfFiles(system_dir+"/"+lp_real)){
        val system = file.getName
        system_sents(system) = WMT11.loadContent(file)
      }
      this.system_sents(lp) = system_sents
    }
  }
  
  private def load_references():Unit={
    for(lang <- langs){
      val lang_really = if(lang eq "cs") "cz" else lang
      val fn = references_dir+"/newstest2011-ref."+lang_really
      val content = WMT11.loadContent(new File(fn))
      for(lp <- lang_pairs){
        if(lp matches s".*-$lang"){
          ref_sents(lp) = content
        }
      }
    }
  }
  
}

object WMT11 {
  
  def loadJudgments(dir:String) : List[Judgment] = {
    val loader = new WMT11(dir)
    loader.load()
    loader.judgments
  }

  private def loadContent(file:File) : Array[String] = {
    Source.fromFile(file, "UTF-8").getLines().toArray
  }
  
  private def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }
  
  private def long_to_short(lang:String) : String = {
    lang match {
      case "English" => "en"
      case "Hindi"   => "hi"
      case "Czech"   => "cs"
      case "Russian" => "ru"
      case "Spanish" => "es"
      case "French"  => "fr"
      case "German"  => "de"
    }
  }

  private def short_to_long(lang:String) : String = {
    lang match {
      case "en"      => "English"
      case "hi"      => "Hindi"  
      case "cs"|"cz" => "Czech"  
      case "ru"      => "Russian"
      case "es"      => "Spanish"
      case "fr"      => "French" 
      case "de"      => "German" 
    }
  }
}
