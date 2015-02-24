#pragma once

#include<cassert>
#include<fstream>

#include "MT.hpp"
#include"ps_defs.hpp"

namespace ParticleSimulator{
    template<class Tptcl>
    class ParticleSystem{
    private:
        //Tptcl * ptcl_;
        ReallocatableArray<Tptcl> ptcl_;
        ReallocatableArray<Tptcl> ptcl_send_;
        ReallocatableArray<Tptcl> ptcl_recv_;
        //S32 n_ptcl_limit_;
        //S32 n_ptcl_;
        S32 n_smp_ptcl_tot_;
        bool first_call_by_initialize;
        bool first_call_by_setAverageTargetNumberOfSampleParticlePerProcess;
        bool first_call_by_DomainInfo_collect_sample_particle;
        /* AT_DEBUG
        inline bool determineWhetherParticleIsInDomain(const F32vec & pos,
                                                       const F32ort & domain) {
        */
        inline bool determineWhetherParticleIsInDomain(const F64vec & pos,
                                                       const F64ort & domain) {
            bool ret = true;
            for(S32 k = 0; k < DIMENSION; k++)
                ret *= (domain.low_[k] <= pos[k]) * (pos[k] < domain.high_[k]);
            return ret;
        }

        /* AT_DEBUG
        S32 searchWhichDomainParticleGoTo(const F32vec & pos,
                                          const S32 n_domain [],
                                          const F32ort domain []) {
        */


        S32 searchWhichDomainParticleGoTo(const F64vec & pos,
                                          const S32 n_domain [],
                                          const F64ort domain []) {
#ifdef PARTICLE_SIMULATOR_TWO_DIMENSION
            S32 idomain = 0;
            const S32 ny = n_domain[1];
            while(domain[idomain].high_[0] <= pos[0])
                idomain += ny;
            while(domain[idomain].high_[1] <= pos[1])
                idomain++;
            return idomain;
#else
            S32 idomain = 0;
            const S32 nynz = n_domain[1] * n_domain[2];
            while(domain[idomain].high_[0] <= pos[0])
                idomain += nynz;
            const S32 nz   = n_domain[2];
            while(domain[idomain].high_[1] <= pos[1])
                idomain += nz;
            while(domain[idomain].high_[2] <= pos[2])
                idomain++;            
            return idomain;
#endif
        }

//        template<class Tvec, class Tdinfo>
//        S32 whereToGo(const Tvec & pos, const Tdinfo & dinfo);

    public:
        ParticleSystem() {
            first_call_by_setAverageTargetNumberOfSampleParticlePerProcess = true;
            first_call_by_initialize = true;
            first_call_by_DomainInfo_collect_sample_particle = true;
            n_smp_ptcl_tot_ = 30 * Comm::getNumberOfProc();
        }
	
        void initialize() {
            assert(first_call_by_initialize);
            first_call_by_initialize = false;
//            first_call_by_DomainInfo_collect_sample_particle = true;            
        }

        void setAverageTargetNumberOfSampleParticlePerProcess(const S32 &nsampleperprocess) {
            assert(first_call_by_setAverageTargetNumberOfSampleParticlePerProcess);
            first_call_by_setAverageTargetNumberOfSampleParticlePerProcess = false;
            n_smp_ptcl_tot_ = nsampleperprocess * Comm::getNumberOfProc();
        }

        S32 getTargetNumberOfSampleParticle() {
            return n_smp_ptcl_tot_;
        }

        bool getFirstCallByDomainInfoCollectSampleParticle() {
            if(first_call_by_DomainInfo_collect_sample_particle) {
                first_call_by_DomainInfo_collect_sample_particle = false;
                return true;
            } else {
                return false;
            }
        }

        void createParticle(const S32 n_limit, bool clear=true){
            //n_ptcl_limit_ = n_limit;
            //ptcl_ = new Tptcl[n_ptcl_limit_];
	    ptcl_.reserve(n_limit);
	    ptcl_.resizeNoInitialize(0);
        }

	
        //void setNumberOfParticleLocal(const S32 n){ n_ptcl_ = n; }
	void setNumberOfParticleLocal(const S32 n){
		//15/02/20 Hosono bug(?) fix.
	    ptcl_.reserve(n*3+1000);
	    ptcl_.resizeNoInitialize(n);
	}
        ////////////////
        // 05/01/30 Hosono From
        ////////////////
        //dummy class for the case if User does NOT define the file header.
        //TO BE PRIVATE
        struct DummyHeader{
            void writeAscii(FILE* fp) const{
            }
            int readAscii (FILE* fp){
                return -1;
            }
            void writeBinary(FILE* fp) const{
            }
            int readBinary (FILE* fp){
                return -1;
            }
        };
        template <class Theader>
        void writeParticleAsciiImpl(const char * const filename, const char * const format, const Theader * const header){
            if(format == NULL){
                #ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
                //declare local # of ptcl.
                const S32 n_ptcl_ = ptcl_.size();
                //get # of process.
                const S32 n_proc = Comm::getNumberOfProc();
                //get # of ptcls in each process.
                S32 *n_ptcl = new S32[n_proc];
                //Gather # of particles.
                MPI::COMM_WORLD.Allgather(&n_ptcl_, 1, GetDataType<S32>(), n_ptcl, 1, GetDataType<S32>());
                //set displacement
                S32 *n_ptcl_displs = new S32[n_proc+1];
                n_ptcl_displs[0] = 0;
                for(S32 i = 0 ; i < n_proc ; ++ i){
                    n_ptcl_displs[i+1] = n_ptcl_displs[i] + n_ptcl[i];
                }
                const S32 n_tot = n_ptcl_displs[n_proc];
                Tptcl *ptcl = new Tptcl[n_tot];
                //gather data
                MPI::COMM_WORLD.Gatherv(ptcl_.getPointer(), n_ptcl_, GetDataType<Tptcl>(), ptcl, n_ptcl, n_ptcl_displs, GetDataType<Tptcl>(), 0);
                if(Comm::getRank() == 0){
                    FILE* fp = fopen(filename, "w");
                    header->writeAscii(fp);
                    for(S32 i = 0 ; i < n_tot ; ++ i){
                        ptcl[i].writeAscii(fp);
                    }
                    fclose(fp);
                }
                delete [] n_ptcl;
                delete [] n_ptcl_displs;
                delete [] ptcl; 
                #else
                const S32 n_tot = ptcl_.size();
                if(Comm::getRank() == 0){
                    FILE* fp = fopen(filename, "w");
                    header->writeAscii(fp);
                    for(S32 i = 0 ; i < n_tot ; ++ i){
                        ptcl_[i].writeAscii(fp);
                    }
                    fclose(fp);
                }
                #endif
            }else{
                char output[256];
                sprintf(output, format, filename, Comm::getNumberOfProc(), Comm::getRank());
                FILE* fp = fopen(output, "w");
                header->writeAscii(fp);
                for(S32 i = 0 ; i < ptcl_.size() ; ++ i){
                    ptcl_[i].writeAscii(fp);
                }
                fclose(fp);
            }
        }
        //write
        template <class Theader>
        void writeParticleAscii(const char * const filename, const char * const format, const Theader& header){
            writeParticleAsciiImpl(filename, format, &header);
        }
        template <class Theader>
        void writeParticleAscii(const char * const filename, const Theader& header){
            writeParticleAsciiImpl(filename, NULL, &header);
        }
        void writeParticleAscii(const char * const filename, const char * format){
            writeParticleAsciiImpl<DummyHeader>(filename, format, NULL);
        }
        void writeParticleAscii(const char * const filename){
            writeParticleAsciiImpl<DummyHeader>(filename, NULL, NULL);
        }
        //read
        template <class Theader>
        void readParticleAsciiImpl(const char * const filename, const char * const format, Theader * const header){
            if(format == NULL){//Read from single file
                if(Comm::getRank() == 0){
                    FILE* fp = fopen(filename, "r");
                    S32 n_ptcl_ = header->readAscii(fp);
                    while('\n' == getc(fp));
                    if(n_ptcl_ < 0){//User does NOT return # of ptcl
                        //count # of lines
                        n_ptcl_ = 0;
                        //KN
                        for(int c ; (c = getc(fp)) != EOF ; n_ptcl_ += '\n' == c ? 1 : 0){}
                        fclose(fp);
                        fp = fopen(filename, "r");
                        header->readAscii(fp);
                        while('\n' == getc(fp));
                    }
                    //Inform the # of ptcl for each process.
                    const S32 n_proc = Comm::getNumberOfProc();
                    S32 *n_ptcl = new S32[n_proc];
                    for(S32 i = 0 ; i < n_proc ; ++ i){
                        n_ptcl[i] = n_ptcl_ / n_proc;
                    }
                    n_ptcl[0] += n_ptcl_ % n_proc;
                    #ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
                    MPI::COMM_WORLD.Scatter(n_ptcl, 1, GetDataType<S32>(), &n_ptcl_, 1, GetDataType<S32>(), 0);
                    #endif
                    //allocate ptcl.
                    //First of all, Rank 0 reads its own particle.
                    this->createParticle(n_ptcl_ << 2);//Magic shift
                    ptcl_.resizeNoInitialize(n_ptcl_);
                    for(int i = 0 ; i < n_ptcl_ ; ++ i){
                        ptcl_[i].readAscii(fp);
                    }
                    //Read remaining data to buffer and send them to appropriate process.
                    #ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
                    for(S32 rank = 1 ; rank < n_proc ; ++ rank){
                        Tptcl * buffer = new Tptcl[n_ptcl[rank]];
                        for(int i = 0 ; i < n_ptcl[rank] ; ++ i){
                            buffer[i].readAscii(fp);
                        }
                        MPI::COMM_WORLD.Send(buffer, n_ptcl[rank], GetDataType<Tptcl>(), rank, 0);
                        delete [] buffer;
                    }
                    #endif
                    //End.
                    delete [] n_ptcl;
                    fclose(fp);
                }else{
                    #ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
                    //Receive the # of ptcl from Rank 0
                    S32 n_ptcl_loc;
                    S32 *n_ptcl = new S32[Comm::getNumberOfProc()];
                    MPI::COMM_WORLD.Scatter(n_ptcl, 1, GetDataType<S32>(), &n_ptcl_loc, 1, GetDataType<S32>(), 0);
                    delete [] n_ptcl;
                    //allocate ptcl.
                    this->createParticle(n_ptcl_loc << 2);//Magic shift
                    ptcl_.resizeNoInitialize(n_ptcl_loc);
                    MPI::COMM_WORLD.Recv(ptcl_.getPointer(), ptcl_.size(), GetDataType<Tptcl>(), 0, 0);
                    #endif
                }
            }else{//Read from multiple file
                char input[256];
                sprintf(input, format, filename, Comm::getNumberOfProc(), Comm::getRank());
                FILE* fp = fopen(input, "r");
                S32 n_ptcl_ = header->readAscii(fp);
                while('\n' == getc(fp));
                if(n_ptcl_ >= 0){
                    //User returns # of ptcl.
                    this->createParticle(n_ptcl_ << 2);//Magic shift
                    ptcl_.resizeNoInitialize(n_ptcl_);
                    for(S32 i = 0 ; i < n_ptcl_ ; ++ i){
                        ptcl_[i].readAscii(fp);
                    }
                    fclose(fp);
                }else{//User does NOT return # of ptcl
                    //count # of lines
                    n_ptcl_ = 0;
                    for(int c ; (c = getc(fp)) != EOF ; n_ptcl_ += c == '\n' ? 1 : 0){}
                    fclose(fp);
                    //
                    FILE* fp = fopen(input, "r");
                    header->readAscii(fp);
                    while('\n' == getc(fp));
                    this->createParticle(n_ptcl_ << 2);//Magic shift
                    ptcl_.resizeNoInitialize(n_ptcl_);
                    for(S32 i = 0 ; i < ptcl_.size() ; ++ i){
                        ptcl_[i].readAscii(fp);
                    }
                    fclose(fp);
                }
                

            }
        }
        template <class Theader>
        void readParticleAscii(const char * const filename, const char * const format, Theader& header){
            readParticleAsciiImpl(filename, format, &header);
        }
        template <class Theader>
        void readParticleAscii(const char * const filename, Theader& header){
            readParticleAsciiImpl(filename, NULL, &header);
        }
        void readParticleAscii(const char * const filename, const char * const format){
            readParticleAsciiImpl<DummyHeader>(filename, format, NULL);
        }
        void readParticleAscii(const char * const filename){
            readParticleAsciiImpl<DummyHeader>(filename, NULL, NULL);
        }
        ////////////////
        // 05/01/30 Hosono To
        ////////////////
        Tptcl & operator [] (const S32 id) {return ptcl_[id];}
        const Tptcl & operator [] (const S32 id) const {return ptcl_[id];}
        Tptcl & getParticle(const S32 id=0) {return ptcl_[id];}
        const Tptcl & getParticle(const S32 id=0) const {return ptcl_[id];}
        Tptcl * getParticlePointer(const S32 id=0) const {return ptcl_+id;}
        //S32 getNumberOfParticleLocal() const {return n_ptcl_;}
        S32 getNumberOfParticleLocal() const {return ptcl_.size();}
        ////////////////
        // 05/02/04 Hosono From
        ////////////////
        S32 getNumberOfParticleGlobal() const {
            #ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
            //get # of process.
            const S32 n_proc = Comm::getNumberOfProc();
            //get # of ptcls in each process.
            S32 *n_ptcl = new S32[n_proc];
            //Gather # of particles.
            S32 n_ptcl_ = ptcl_.size();
            MPI::COMM_WORLD.Allgather(&n_ptcl_, 1, GetDataType<S32>(), n_ptcl, 1, GetDataType<S32>());
            //set displacement
            S32 *n_ptcl_displs = new S32[n_proc+1];
            n_ptcl_displs[0] = 0;
            for(S32 i = 0 ; i < n_proc ; ++ i){
                n_ptcl_displs[i+1] = n_ptcl_displs[i] + n_ptcl[i];
            }
            return n_ptcl_displs[n_proc];
            #else
            return ptcl_.size();
            #endif
        }
        ////////////////
        // 05/02/04 Hosono To
        ////////////////

        F64 getHalfLength(const F64vec & center=0.0){
            F64 hl_max_loc = (ptcl_[0].getPos() - center).applyEach(Abs<F64>()).getMax();
            const S32 n_ptcl = ptcl_.size();
            for(size_t i=1; i<n_ptcl; i++){
                F64 hl_tmp = (ptcl_[i].getPos() - center).applyEach(Abs<F64>()).getMax();
                hl_max_loc = (hl_max_loc > hl_tmp) ? hl_max_loc : hl_tmp;
            }
#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
            F64 hl_max_glb;
            MPI::COMM_WORLD.Allreduce(&hl_max_loc, &hl_max_glb, 1, GetDataType<F64>(), MPI::MAX);
            return hl_max_glb;
#else
            return hl_max_loc;
#endif
        }

/*
        F32ort getParticleDomain() {
            F32ort domain;
            
            S32 ndim = DIMENSION;
            S32 nloc = n_ptcl_;
            
            for(S32 k = 0; k < ndim; k++) {
                domain.low_[k]  = ptcl_[0].getPos[k];
                domain.high_[k] = ptcl_[0].getPos[k];
            }
            
            for(S32 i = 0; i < nloc; i++) {
                for(S32 k = 0; k < ndim; k++) {
                    F32 pos = ptcl_[i].getPos[k];
                    if(pos < domain.low_[k])
                        domain.low_[k]  = pos;
                    if(pos > domain.high_[k])
                        domain.high_[k] = pos;
                }
            }

            return domain;
            
        }
*/

// *******************************************************************        
// ************** This can be replaced with MT method. ***************
// *******************************************************************        
        inline F64 frand() {
//                return (double) rand() / ((double) RAND_MAX + 1.);
            //return genrand_res53();
            return MT::genrand_res53();
        }
// *******************************************************************        
// *******************************************************************        

        inline S32 getUniformDistributionFromArg1ToArg2(S32 arg1,
                                                        S32 arg2) {
            S32 random_number;
            
            random_number = (S32)((arg2 - arg1 + 1) * frand()) + arg1;
            
            return random_number;
        }

        /* AT_DEBU
        void getSampleParticle(S32 & number_of_sample_particle,
                               F32vec pos_sample[],
                               const F32 weight=1.0) {
        */
        void getSampleParticle(S32 & number_of_sample_particle,
                               F64vec pos_sample[],
                               const F32 weight=1.0) {

#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
/*
                S32 nglb = 0;
                S32 nloc = n_ptcl_;

                MPI::COMM_WORLD.Allreduce(&nloc, &nglb, 1, MPI::INT, MPI::SUM);
                
                number_of_sample_particle = (S32)(nloc * n_smp_ptcl_tot_ / (F32)nglb);
                number_of_sample_particle = (number_of_sample_particle < nloc) ? number_of_sample_particle : nloc;
*/

            F32 weight_all = 0.;
            //S32 nloc = n_ptcl_;
            S32 nloc = ptcl_.size();
            //MPI::COMM_WORLD.Allreduce(&weight, &weight_all, 1, MPI::FLOAT, MPI::SUM);
            MPI::COMM_WORLD.Allreduce(&weight, &weight_all, 1, GetDataType<F32>(), MPI::SUM);
            number_of_sample_particle = (S32)(weight * n_smp_ptcl_tot_ / weight_all);
#if 1
// modified to limit # of sample particles by M.I. 
            const F32 coef_limitter = 0.2;
            S64 nglb = Comm::getSum( (S64)nloc );
            S32 number_of_sample_particle_limit = ((S64)nloc * n_smp_ptcl_tot_) / (nglb * (1.0 + coef_limitter)); // lower limit
            //std::cerr<<"number_of_sample_particle_limit="<<number_of_sample_particle_limit<<std::endl;
            //std::cerr<<"number_of_sample_particle="<<number_of_sample_particle<<std::endl;
            number_of_sample_particle = (number_of_sample_particle > number_of_sample_particle_limit) ? number_of_sample_particle : number_of_sample_particle_limit;
#endif
            number_of_sample_particle = (number_of_sample_particle < nloc) ? number_of_sample_particle : nloc;

            S32 *record = new S32[number_of_sample_particle];
            for(S32 i = 0; i < number_of_sample_particle; i++) {
                    S32 j = getUniformDistributionFromArg1ToArg2(i, nloc-1);
                    Tptcl hold = ptcl_[j];
                    ptcl_[j]   = ptcl_[i];
                    ptcl_[i]   = hold;        
                    record[i]  = j;
                }
                for(S32 i = 0; i < number_of_sample_particle; i++) {
                    pos_sample[i] = ptcl_[i].getPos();
                }

                for(S32 i = number_of_sample_particle - 1; i >= 0; i--) {
                    S32 j = record[i];
                    Tptcl hold = ptcl_[j];
                    ptcl_[j]   = ptcl_[i];
                    ptcl_[i]   = hold;
                }

                delete [] record;
                
                return;
#endif
            }

        template<class Tdinfo>
        void exchangeParticle(Tdinfo & dinfo) {
#ifdef PARTICLE_SIMULATOR_MPI_PARALLEL
            //const S32 nloc  = n_ptcl_;
	    const S32 nloc  = ptcl_.size();
            const S32 rank  = MPI::COMM_WORLD.Get_rank();
            const S32 nproc = MPI::COMM_WORLD.Get_size();
            const S32 * n_domain = dinfo.getPointerOfNDomain();

            /* AT_DEBUG
            F32ort * pos_domain = dinfo.getPointerOfPosDomain();
            F32ort thisdomain = dinfo.getPosDomain(rank);
            */
            const F64ort * pos_domain = dinfo.getPointerOfPosDomain();
            const F64ort thisdomain = dinfo.getPosDomain(rank);

            S32 nsendtot = 0;
            S32 *nsend0  = new S32[nproc];
            S32 *nsend1  = new S32[nproc];
            S32 nrecvtot = 0;
            S32 *nrecv0  = new S32[nproc];
            S32 *nrecv1  = new S32[nproc];
            for(S32 i = 0; i < nproc; i++) {
                nsend0[i] = 0;
                nsend1[i] = 0;
                nrecv0[i] = 0;
                nrecv1[i] = 0;
            }
            MPI::Request * req_send = new MPI::Request[nproc];
            MPI::Request * req_recv = new MPI::Request[nproc];
            // *** count the number of send particles preliminary *
            for(S32 ip = 0; ip < nloc; ip++) {
                if( dinfo.getPosRootDomain().notOverlapped(ptcl_[ip].getPos()) ){
                    //std::cerr<<"ptcl_[ip].getPos()="<<ptcl_[ip].getPos()<<std::endl;
                    //std::cerr<<"dinfo.getPosRootDomain()="<<dinfo.getPosRootDomain()<<std::endl;
                    //throw"PS_ERROR: position of particle is out of root domain";
                    PARTICLE_SIMULATOR_PRINT_ERROR("position of particle is out of root domain");
                    std::cerr<<"ptcl_[ip].getPos()="<<ptcl_[ip].getPos()<<std::endl;
                    std::cerr<<"dinfo.getPosRootDomain()="<<dinfo.getPosRootDomain()<<std::endl;
                    Abort(-1);
                }
                if(!determineWhetherParticleIsInDomain(ptcl_[ip].getPos(), thisdomain)) {
                    S32 srank = searchWhichDomainParticleGoTo(ptcl_[ip].getPos(), n_domain, pos_domain);
                    nsend1[srank]++;
                    nsendtot++;
                }
            }
            nsend0[0] = 0;
            for(S32 i = 1; i < nproc; i++) {
                nsend0[i] += nsend0[i-1] + nsend1[i-1];
            }
            //ptcl_send_ = new Tptcl[nsendtot];
            ptcl_send_.resizeNoInitialize(nsendtot);
            // ****************************************************
            // *** align send particles on ptcl_send_ *************
            //std::cerr<<"check 0"<<std::endl;
            for(S32 i = 0; i < nproc; i++)
                nsend1[i] = 0;
            S32 iloc = 0;
            for(S32 ip = 0; ip < nloc; ip++) {
                if(determineWhetherParticleIsInDomain(ptcl_[ip].getPos(), thisdomain)) {
                    ptcl_[iloc] = ptcl_[ip];
                    iloc++;
                } else {
                    S32 srank = searchWhichDomainParticleGoTo(ptcl_[ip].getPos(), n_domain, pos_domain);
                    S32 jloc = nsend0[srank] + nsend1[srank];
                    ptcl_send_[jloc] = ptcl_[ip];
                    nsend1[srank]++;
                }
            }           
            //n_ptcl_ = iloc;
	    //std::cerr<<"check 1"<<std::endl;
	    ptcl_.resizeNoInitialize(iloc);
	    //std::cerr<<"ptcl_.size()="<<ptcl_.size()<<std::endl;
/*
            if(rank == 0) {
                char filename[1024];
                FILE *fp;
                sprintf(filename, "out/send_%04d_%04d.txt", rank, rank);
                fp = fopen(filename, "w");
                for(S32 j = 0; j < n_ptcl_; j++)
                    fprintf(fp, "%+e %+e %+e\n",
                            ptcl_[j].getPos()[0],
                            ptcl_[j].getPos()[1],
                            ptcl_[j].getPos()[2]);
                fclose(fp);
                for(S32 i = 1; i < nproc; i++) {
                    S32 srank = (rank + i) % nproc;
                    sprintf(filename, "out/send_%04d_%04d.txt", rank, srank);
                    fp = fopen(filename, "w");
                    S32 next = (srank + 1 < nproc) ? nsend0[srank+1] : nsendtot;
                    for(S32 j = nsend0[srank]; j < next; j++)
                        fprintf(fp, "%+e %+e %+e\n",
                                ptcl_send_[j].getPos()[0],
                                ptcl_send_[j].getPos()[1],
                                ptcl_send_[j].getPos()[2]);
                    fclose(fp);
                }
            }
*/
            // ****************************************************
            // *** receive the number of receive particles ********
            MPI::COMM_WORLD.Alltoall(nsend1, 1, GetDataType<S32>(), nrecv1, 1, GetDataType<S32>());
	    //std::cerr<<"check 2"<<std::endl;
            for(S32 i = 0; i < nproc; i++)
                nrecvtot += nrecv1[i];
            //assert(n_ptcl_ + nrecvtot <= n_ptcl_limit_);
	    //ptcl_.reserve(n_ptcl_ + nrecvtot);
	    //ptcl_.resizeNoInitialize(n_ptcl_ + nrecvtot);
            //ptcl_recv_ = new Tptcl[nrecvtot];
	    ptcl_recv_.resizeNoInitialize(nrecvtot);
	    //std::cerr<<"check 3"<<std::endl;
/*
            {
                char filename[1024];
                FILE *fp;
                sprintf(filename, "out/send_%04d.txt", rank);
                fp = fopen(filename, "w");
                fprintf(fp, "%d", rank);
                for(S32 i = 0; i < nproc; i++) {
                    fprintf(fp, "%6d", nsend1[i]);
                }
                fprintf(fp, "\n");
                fclose(fp);
                sprintf(filename, "out/recv_%04d.txt", rank);
                fp = fopen(filename, "w");
                fprintf(fp, "%d", rank);
                for(S32 i = 0; i < nproc; i++) {
                    fprintf(fp, "%6d", nrecv[i]);
                }
                fprintf(fp, "\n");
                fclose(fp);
            }
*/
            // ****************************************************
            // *** send and receive particles *********************
            nrecv0[0] = 0;
            for(S32 i = 1; i < nproc; i++) {
                nrecv0[i] += nrecv0[i-1] + nrecv1[i-1];
            }
            S32 nsendnode = 0;
            S32 nrecvnode = 0;
	    //std::cerr<<"check 4"<<std::endl;
            for(S32 ib = 1; ib < nproc; ib++) {
                S32 idsend = (ib + rank) % nproc;
                if(nsend1[idsend] > 0) {
                    S32 adrsend = nsend0[idsend];                    
                    S32 tagsend = (rank < idsend) ? rank : idsend;                    
                    //req_send[nsendnode] = MPI::COMM_WORLD.Isend(ptcl_send_+adrsend, nsend1[idsend]*sizeof(Tptcl), MPI::BYTE, idsend, tagsend);
		    req_send[nsendnode] = MPI::COMM_WORLD.Isend(ptcl_send_.getPointer(adrsend), nsend1[idsend], GetDataType<Tptcl>(), idsend, tagsend);
                    nsendnode++;
                }                
                S32 idrecv = (nproc + rank - ib) % nproc;
                if(nrecv1[idrecv] > 0) {
                    S32 adrrecv = nrecv0[idrecv];
                    S32 tagrecv = (rank < idrecv) ? rank : idrecv;
                    //req_recv[nrecvnode] = MPI::COMM_WORLD.Irecv(ptcl_recv_+adrrecv, nrecv1[idrecv]*sizeof(Tptcl), MPI::BYTE, idrecv, tagrecv);
		    req_recv[nrecvnode] = MPI::COMM_WORLD.Irecv(ptcl_recv_.getPointer(adrrecv), nrecv1[idrecv], GetDataType<Tptcl>(), idrecv, tagrecv);
                    nrecvnode++;
                }
            }
            MPI::Request::Waitall(nsendnode, req_send);
            MPI::Request::Waitall(nrecvnode, req_recv);
	    //std::cerr<<"check 5"<<std::endl;
            // ****************************************************            
            // *** align particles ********************************
	    /*
            for(S32 ip = 0; ip < nrecvtot; ip++) {
                ptcl_[n_ptcl_] = ptcl_recv_[ip];
                n_ptcl_++;
            }
	    */
	    ptcl_.reserve( ptcl_.size()+nrecvtot );
	    //ptcl_.dump("dump ptcl");
	    for(S32 ip = 0; ip < nrecvtot; ip++) {
                ptcl_.pushBackNoCheck(ptcl_recv_[ip]);
            }
	    //std::cerr<<"ptcl_.size()="<<ptcl_.size()<<std::endl;
            // ****************************************************            

            delete [] nsend0;
            delete [] nsend1;
            delete [] nrecv0;
            delete [] nrecv1;
            delete [] req_send;
            delete [] req_recv;
            //delete [] ptcl_send_;
            //delete [] ptcl_recv_;
#endif
        }

        // for DEBUG functions
        template<class Treal, class Tvec>
        void calcCMDirect(Treal & mass_cm, Tvec & pos_cm){
            mass_cm = 0.0;
            pos_cm = 0.0;
	    const S32 n_ptcl = ptcl_.size();
            for(S32 i=0; i<n_ptcl; i++){
                mass_cm += ptcl_[i].mass;
                pos_cm += ptcl_[i].mass * ptcl_[i].pos;
            }
            pos_cm /= mass_cm;
        }

        bool checkExchangeParticleAllParticleInside(DomainInfo & dinfo);
        bool checkExchangeParticleSumOfNumberOfParticle(DomainInfo & dinfo,
                                                        S32 ntot_init);

        void adjustPositionIntoRootDomain(const DomainInfo & dinfo){
            const F64ort pos_root = dinfo.getPosRootDomain();
            const F64vec len_root = pos_root.getFullLength();
            const S32 n = ptcl_.size();
#pragma omp parallel for
            for(size_t i=0; i<n; i++){
                F64vec pos_new = ptcl_[i].getPos() ;
                if( pos_root.notOverlapped(pos_new) ){
                    while(pos_new.x < pos_root.low_.x){
                        pos_new.x += len_root.x;
                    }
                    while(pos_new.x > pos_root.high_.x){
                        pos_new.x -= len_root.x;
                    }
                    if(pos_new.x == pos_root.high_.x){
                        pos_new.x = pos_root.low_.x;
                    }
                    while(pos_new.y < pos_root.low_.y){
                        pos_new.y += len_root.y;
                    }
                    while(pos_new.y >= pos_root.high_.y){
                        pos_new.y -= len_root.y;
                    }
                    if(pos_new.y == pos_root.high_.y){
                        pos_new.y = pos_root.low_.y;
                    }
#ifndef PARTICLE_SIMULATOR_TWO_DIMENSION
                    while(pos_new.z < pos_root.low_.z){
                        pos_new.z += len_root.z;
                    }
                    while(pos_new.z >= pos_root.high_.z){
                        pos_new.z -= len_root.z;
                    }
                    if(pos_new.z == pos_root.high_.z){
                        pos_new.z = pos_root.low_.z;
                    }
#endif
                }
                ptcl_[i].setPos(pos_new);
            }
        }

        size_t getMemSizeUsed() const {
            return ptcl_.getMemSize() + ptcl_send_.getMemSize() + ptcl_recv_.getMemSize();
        }
	
    };
}
