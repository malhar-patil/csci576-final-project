#pragma once 

#include<vector>
#include<QFile>
#include<QDir>
#include<QTextStream>
#include<QFileInfo>
#include "SegmentData.h"

namespace SegLoader {

inline std::vector<Segment> load(const QString& videoPath){
    QFileInfo info(videoPath);
    // Prefer merged/fused result; fall back to raw <stem>.seg if not present.
    QString mergedSegPath = info.dir().filePath(info.completeBaseName()+"_merged_final.seg");
    QString rawSegPath = info.dir().filePath(info.completeBaseName()+".seg");
    QString segPath = QFile::exists(mergedSegPath) ? mergedSegPath : rawSegPath;

    if(!QFile::exists(segPath)){
        return {};
    }

    QFile segFile(segPath);
    if(!segFile.open(QIODevice::ReadOnly | QIODevice::Text)){
        return {};
    }

    QTextStream in(&segFile);
    QString allText = in.readAll();
    QStringList lines = allText.split("\n");

    double duration = 0.0;
    std::vector<Segment> segments;

    for(const QString& line : lines){
        
        if(line.contains("duration_seconds") && !line.contains("start_seconds")){
            int colon = line.trimmed().indexOf(':');
            QString val = line.trimmed().mid(colon+1);
            val = val.replace(",","");
            duration = val.toDouble();
        }
        else if(line.contains("start_seconds")){
            int colon = line.trimmed().indexOf(':');
            QString val = line.trimmed().mid(colon+1);
            val = val.replace(",","");
            Segment newSegment;
            newSegment.start_sec = val.toDouble();
            segments.push_back(newSegment);
        }
        else if(line.contains("category") && !segments.empty()){
            int colon = line.trimmed().indexOf(':');
            QString val = line.trimmed().mid(colon+1);
            val = val.replace(",", "").trimmed().toUpper();
            if(val == "CONTENT"){
                segments.back().type = SegmentType::content;
            }
            else{
                segments.back().type = SegmentType::non_content;
            }
        }
    }

    for(int i=0; i<segments.size()-1;i++){
        segments[i].end_sec = segments[i+1].start_sec;
    }
    if(!segments.empty()){
        segments.back().end_sec = duration;
    }
    return segments;
}

}

