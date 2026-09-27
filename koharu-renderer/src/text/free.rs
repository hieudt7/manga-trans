//! FREE_TEXT space and paragraph search. No source mask enters this module.
use std::collections::VecDeque;

use anyhow::Result;
use image::{GrayImage, Luma, Rgb, RgbImage};
use imageproc::{distance_transform::Norm, morphology::erode};
use koharu_types::TextBlock;
use serde::Serialize;

use crate::{font::Font, layout::{LayoutRun, TextLayout, WritingMode},
    renderer::{RenderOptions, RenderStrokeOptions, TinySkiaRenderer}, text::latin::LayoutBox};

struct Region {
    mask: GrayImage,
    tone: u8,
    area: usize,
}

#[derive(Clone, Serialize)]
pub struct LineChoice {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub available_width: f32,
    #[serde(skip)]
    start: usize,
    #[serde(skip)]
    end: usize,
}

#[derive(Clone, Serialize)]
pub struct Candidate {
    pub region: usize,
    pub font_size: f32,
    pub line_spacing: f32,
    pub score: f32,
    pub break_cost: f32,
    pub size_penalty: f32,
    pub proximity_penalty: f32,
    pub shape_penalty: f32,
    pub lines: Vec<LineChoice>,
}

pub struct Selection<'a> {
    pub layout: LayoutRun<'a>,
    pub bounds: LayoutBox,
    pub selected: Candidate,
    candidates: Vec<Candidate>,
    regions: Vec<Region>,
    crop: [u32;4],
    step: u32,
    evaluated: usize,
    source: [f32;4],
    source_size: f32,
}

/// Background modes come from the restored image, not a dark/light assumption.
/// Eroded connected components prevent crossing outlines, gutters, or thin
/// channels through hair and speed lines. Search radius is measured in glyphs.
fn regions(page: &GrayImage, block: &TextBlock, glyph: f32) -> (Vec<Region>, [u32;4], u32) {
    let cx = block.x+block.width*0.5;
    let cy = block.y+block.height*0.5;
    let rx = glyph*7.0;
    let ry = glyph*5.0;
    let x0 = (cx-rx).max(0.0) as u32;
    let y0 = (cy-ry).max(0.0) as u32;
    let x1 = (cx+rx).min(page.width() as f32) as u32;
    let y1 = (cy+ry).min(page.height() as f32) as u32;
    let step = (glyph*0.07).round().max(2.0) as u32;
    let (w,h) = ((x1-x0)/step, (y1-y0)/step);
    let crop = [x0,y0,w*step,h*step];
    if w == 0 || h == 0 { return (Vec::new(),crop,step); }
    let mut hist = [0usize;16];
    for y in (block.y.max(0.0) as u32..((block.y+block.height) as u32).min(page.height())).step_by(step as usize) {
        for x in (block.x.max(0.0) as u32..((block.x+block.width) as u32).min(page.width())).step_by(step as usize) {
            hist[(page.get_pixel(x,y)[0]/16) as usize] += 1;
        }
    }
    let mut ranked: Vec<usize> = (0..16).collect();
    ranked.sort_by_key(|i| std::cmp::Reverse(hist[*i]));
    let mut modes: Vec<usize> = Vec::new();
    for bin in ranked {
        if hist[bin] * 10 < hist.iter().sum::<usize>() { continue; }
        if modes.iter().all(|other| other.abs_diff(bin) >= 3) { modes.push(bin); }
        if modes.len() == 2 { break; }
    }
    let mut found = Vec::new();
    for bin in modes {
        let tone = (bin*16+8).min(255) as u8;
        let raw = GrayImage::from_fn(w,h,|gx,gy| {
            let mut lo = 255u8;
            let mut hi = 0u8;
            for y in y0+gy*step..y0+(gy+1)*step { for x in x0+gx*step..x0+(gx+1)*step {
                let v=page.get_pixel(x,y)[0]; lo=lo.min(v); hi=hi.max(v);
            }}
            // Both local texture and colour compatibility matter. A bright
            // face is not traversable from a dark background, or vice versa.
            Luma([if hi-lo <= 36 && lo.abs_diff(tone)<=36 && hi.abs_diff(tone)<=36 {255} else {0}])
        });
        let safe = erode(&raw,Norm::LInf,(glyph*0.12/step as f32).ceil().max(1.0) as u8);
        let mut visited=vec![false;(w*h) as usize];
        for sy in 0..h { for sx in 0..w {
            if visited[(sy*w+sx) as usize] || safe.get_pixel(sx,sy)[0]==0 {continue;}
            let mut queue=VecDeque::from([(sx,sy)]);
            let mut points=Vec::new();
            let mut near_source=false;
            visited[(sy*w+sx) as usize]=true;
            while let Some((x,y))=queue.pop_front() {
                points.push((x,y));
                let px=(x0+x*step) as f32; let py=(y0+y*step) as f32;
                near_source |= px>=block.x && px<=block.x+block.width && py>=block.y && py<=block.y+block.height;
                for (nx,ny) in [(x as i32-1,y as i32),(x as i32+1,y as i32),(x as i32,y as i32-1),(x as i32,y as i32+1)] {
                    if nx<0 || ny<0 || nx>=w as i32 || ny>=h as i32 {continue;}
                    let i=(ny as u32*w+nx as u32) as usize;
                    if !visited[i] && safe.get_pixel(nx as u32,ny as u32)[0]!=0 {
                        visited[i]=true;queue.push_back((nx as u32,ny as u32));
                    }
                }
            }
            let area=points.len()*(step*step) as usize;
            if !near_source || (area as f32)<glyph*glyph*5.0 {continue;}
            let mut mask=GrayImage::new(w,h);
            for (x,y) in points {mask.put_pixel(x,y,Luma([255]));}
            found.push(Region{mask,tone,area});
        }}
    }
    found.sort_by_key(|r| std::cmp::Reverse(r.area));
    found.truncate(4);
    (found,crop,step)
}

/// Intersect *every* row a rendered line occupies. Pick a continuous span,
/// never the union of disconnected islands on either side of a drawing.
fn band(region: &Region, crop: [u32;4], step: u32, y: f32, height: f32, cx: f32) -> Option<(f32,f32)> {
    if y<crop[1] as f32 || y+height>(crop[1]+crop[3]) as f32 {return None;}
    let a=((y-crop[1] as f32)/step as f32).floor() as u32;
    let b=((y+height-crop[1] as f32)/step as f32).ceil() as u32;
    let mut start=None;
    let mut best=None;
    let mut best_score=f32::NEG_INFINITY;
    for x in 0..=region.mask.width() {
        let open=x<region.mask.width() && (a..b).all(|row| region.mask.get_pixel(x,row)[0]!=0);
        if open && start.is_none() {start=Some(x);}
        if !open && let Some(s)=start.take() {
            let left=(crop[0]+s*step) as f32;
            let width=((x-s)*step) as f32;
            let score=width-0.15*(left+width*0.5-cx).abs();
            if score>best_score {best=Some((left,width));best_score=score;}
        }
    }
    best
}

pub fn search<'a>(page: &GrayImage, block: &TextBlock, text: &str, font: &'a Font,
    fallbacks: &'a [Font], source_size: f32, ceiling: f32) -> Result<Option<Selection<'a>>> {
    let (regions,crop,step)=regions(page,block,source_size);
    if regions.is_empty() {return Ok(None);}
    let words:Vec<&str>=text.split_whitespace().collect();
    let n=words.len();
    if n==0 {return Ok(None);}
    // Shape each possible whole-word line once. Searching never splits a word
    // or inherits source CJK line breaks. Font units scale linearly.
    const MEASURE:f32=32.0;
    let mut shaped=vec![vec![None;n+1];n];
    for i in 0..n {for j in i+1..=n {
        shaped[i][j]=Some(TextLayout::new(font,None).with_fallback_fonts(fallbacks)
            .run_whole_at(&words[i..j].join(" "),MEASURE)?);
    }}
    let cx=block.x+block.width*0.5; let cy=block.y+block.height*0.5;
    let mut candidates=Vec::new();
    let mut evaluated=0;
    let min_size=(ceiling*0.65).max(page.height() as f32*0.012).ceil() as u32;
    for size in (min_size..=ceiling.ceil() as u32).step_by(2) {
        let size=size as f32; let scale=size/MEASURE;
        let line_height=shaped.iter().flat_map(|r|r.iter().flatten()).map(|l|l.height*scale).fold(size,f32::max);
        for spacing in [1.08,1.18,1.28] {
            let pitch=line_height*spacing;
            for (ri,region) in regions.iter().enumerate() {
                for top in (crop[1]..crop[1]+crop[3]).step_by((source_size*0.18).max(3.0) as usize) {
                    let mut bands=Vec::new();
                    for line in 0..n.min(12) {
                        let y=top as f32+line as f32*pitch;
                        let Some((x,width))=band(region,crop,step,y,line_height,cx) else {break;};
                        if width<size*4.5 {break;}
                        bands.push((x,y,width));
                    }
                    if bands.is_empty() {continue;}
                    evaluated+=1;
                    let mut cost=vec![vec![f32::INFINITY;n+1];bands.len()+1];
                    let mut previous=vec![vec![0usize;n+1];bands.len()+1];
                    cost[0][0]=0.0;
                    for (l,&(_,_,available)) in bands.iter().enumerate() {
                        for i in 0..n {
                            if !cost[l][i].is_finite() {continue;}
                            for j in i+1..=n {
                                let width=shaped[i][j].as_ref().unwrap().width*scale;
                                if width>available {break;}
                                let fill=width/available;
                                let short=(0.6-fill).max(0.0);
                                let singleton=if j-i==1 && n>3 {8.0} else {0.0};
                                let break_cost=1.8+2.0*(1.0-fill).powi(2)+10.0*short.powi(2)+singleton;
                                let v=cost[l][i]+break_cost;
                                if v<cost[l+1][j] {cost[l+1][j]=v; previous[l+1][j]=i;}
                            }
                        }
                    }
                    for count in 1..=bands.len() {
                        if !cost[count][n].is_finite() {continue;}
                        let mut end=n; let mut lines=Vec::new();
                        for line in (0..count).rev() {
                            let start=previous[line+1][end];
                            let (x,y,available)=bands[line];
                            let run=shaped[start][end].as_ref().unwrap();
                            let width=run.width*scale;
                            lines.push(LineChoice{text:words[start..end].join(" "),x:x+(available-width)*0.5,y,
                                width,height:run.height*scale,available_width:available,start,end});
                            end=start;
                        }
                        lines.reverse();
                        let mean_width=lines.iter().map(|l|l.width).sum::<f32>()/count as f32;
                        let balance=lines.windows(2).map(|v|((v[0].width-v[1].width)/mean_width).powi(2)).sum::<f32>();
                        let height=(count-1) as f32*pitch+line_height;
                        let pcx=lines.iter().map(|l|l.x+l.width*0.5).sum::<f32>()/count as f32;
                        let pcy=top as f32+height*0.5;
                        let proximity_penalty=0.8*((pcx-cx).hypot(pcy-cy)/source_size).powi(2);
                        let size_penalty=35.0*(1.0-size/ceiling).powi(2);
                        let shape_penalty=4.0*(height/mean_width-1.0).max(0.0).powi(2)+balance*3.0;
                        let score=cost[count][n]+size_penalty+proximity_penalty+shape_penalty;
                        candidates.push(Candidate{region:ri,font_size:size,line_spacing:spacing,score,
                            break_cost:cost[count][n],size_penalty,proximity_penalty,shape_penalty,lines});
                    }
                }
            }
        }
    }
    candidates.sort_by(|a,b|a.score.total_cmp(&b.score));
    // Preserve different sizes/placements for visual comparison, not eight
    // nearly identical samples of the same paragraph.
    let mut best:Vec<Candidate>=Vec::new();
    for candidate in candidates {
        if best.iter().any(|b|b.region==candidate.region && b.font_size==candidate.font_size
            && (b.lines[0].y-candidate.lines[0].y).abs()<source_size*0.5) {continue;}
        best.push(candidate); if best.len()==8 {break;}
    }
    let Some(selected)=best.first().cloned() else {return Ok(None);};
    let (layout,bounds)=build_layout(&selected,&shaped,MEASURE);
    Ok(Some(Selection{layout,bounds,selected,candidates:best,regions,crop,step,evaluated,
        source:[block.x,block.y,block.width,block.height],source_size}))
}

fn build_layout<'a>(candidate:&Candidate,shaped:&[Vec<Option<LayoutRun<'a>>>],measure:f32)->(LayoutRun<'a>,LayoutBox) {
    let x=candidate.lines.iter().map(|l|l.x).fold(f32::INFINITY,f32::min)-2.0;
    let y=candidate.lines[0].y-2.0;
    let right=candidate.lines.iter().map(|l|l.x+l.width).fold(0.0,f32::max)+2.0;
    let bottom=candidate.lines.iter().map(|l|l.y+l.height).fold(0.0,f32::max)+2.0;
    let scale=candidate.font_size/measure;
    let mut lines=Vec::new();
    for choice in &candidate.lines {
        let mut line=shaped[choice.start][choice.end].as_ref().unwrap().lines[0].clone();
        line.baseline=(choice.x-x+line.baseline.0*scale,choice.y-y+line.baseline.1*scale);
        line.advance*=scale; line.span=Some((choice.x-x,choice.available_width));
        for g in &mut line.glyphs {g.x_advance*=scale;g.y_advance*=scale;g.x_offset*=scale;g.y_offset*=scale;}
        lines.push(line);
    }
    (LayoutRun{lines,width:right-x,height:bottom-y,font_size:candidate.font_size,fits:true,max_word_cuts:0},
        LayoutBox{x,y,width:right-x,height:bottom-y})
}

impl Selection<'_> {
    pub fn debug(&self,id:&str,page:&GrayImage,font:&Font,fallbacks:&[Font],color:[u8;4],stroke:Option<RenderStrokeOptions>,source_stroke:f32)->Result<()> {
        let Ok(root)=std::env::var("KOHARU_DEBUG_FREE_LAYOUT") else {return Ok(());};
        let dir=std::path::Path::new(&root).join(id.replace(['/', '\\'],"_"));
        std::fs::create_dir_all(&dir)?;
        let [x,y,w,h]=self.crop;
        let base=image::imageops::crop_imm(page,x,y,w,h).to_image();
        let rgb=RgbImage::from_fn(w,h,|x,y|Rgb([base.get_pixel(x,y)[0];3]));
        let region=&self.regions[self.selected.region];
        let mask=image::imageops::resize(&region.mask,w,h,image::imageops::FilterType::Nearest);
        mask.save(dir.join("05_available_region.png"))?;
        let mut overlay=rgb.clone();
        for (x,y,p) in overlay.enumerate_pixels_mut() {if mask.get_pixel(x,y)[0]!=0 {p[1]=(u16::from(p[1])/2+110).min(255) as u8;}}
        overlay.save(dir.join("06_available_region_overlay.png"))?;
        let mut sheet=RgbImage::new(w*2,h*2);
        for (index,candidate) in self.candidates.iter().take(4).enumerate() {
            let mut preview=rgb.clone();
            for line in &candidate.lines {
                let layout=TextLayout::new(font,None).with_fallback_fonts(fallbacks).run_whole_at(&line.text,candidate.font_size)?;
                let scaled_stroke=stroke.map(|mut s|{s.width_px*=candidate.font_size/self.selected.font_size;s});
                let rendered=TinySkiaRenderer::new()?.render(&layout,WritingMode::Horizontal,&RenderOptions{font_size:candidate.font_size,color,stroke:scaled_stroke,..Default::default()})?;
                let mut rgba=image::DynamicImage::ImageRgb8(preview).to_rgba8();
                image::imageops::overlay(&mut rgba,&rendered,(line.x-x as f32) as i64,(line.y-y as f32) as i64);
                preview=image::DynamicImage::ImageRgba8(rgba).to_rgb8();
            }
            if index==0 {preview.save(dir.join("08_selected_layout.png"))?;}
            image::imageops::replace(&mut sheet,&preview,(index%2) as i64*w as i64,(index/2) as i64*h as i64);
        }
        sheet.save(dir.join("07_layout_candidates.png"))?;
        let geometry:Vec<_>=self.regions.iter().enumerate().map(|(id,r)|serde_json::json!({
            "id":id,"background_tone":r.tone,"area":r.area,"connected_components":1,
        })).collect();
        let report=serde_json::json!({"source_bbox":self.source,"debug_crop":self.crop,
            "source_character_size":self.source_size,"source_stroke_width":source_stroke,
            "target_stroke_width":stroke.map(|s|s.width_px),"available_regions":geometry,
            "evaluated_placements":self.evaluated,"selected":self.selected,"top_candidates":self.candidates,
            "score_explanation":"Lower is better. Invalid/overflow/edge-crossing candidates rejected. Score = whole-word break cost + relative font-size penalty + distance from source center + tall-shape/adjacent-width imbalance penalties."});
        std::fs::write(dir.join("layout.json"),serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    }
}
