bodylistdat = fopen("nbodylist.dat");
bodylist = textscan(bodylistdat,"%s");
fclose(bodylistdat);

fext = ".dat";
for i = 1:length(bodylist{1})
    bodycoordsdat = fopen(strcat(bodylist{1}{i},fext));
    bodycoords = textscan(bodycoordsdat,"%f %f %f %f %f %f");
    fclose(bodycoordsdat);
    h = plot(bodycoords{1},bodycoords{2},"DisplayName",bodylist{1}{i});
    hold on
end

legend("show")

